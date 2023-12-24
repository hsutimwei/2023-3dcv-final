import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
import torch

class SimpleVO:
	def __init__(self, args):
		camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
		self.K = camera_params['K']
		
		self.dist = camera_params['dist']

		self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))

	def run(self):
		RATIO = 1
		def create_camera():
			points = (np.array([[0, 0, 0], [1, 1, -2], [1, -1, -2], [-1, -1, -2], [-1, 1, -2]]) * RATIO).tolist()
			lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
			lineset = o3d.geometry.LineSet()
			lineset.points = o3d.utility.Vector3dVector(points)
			lineset.lines = o3d.utility.Vector2iVector(lines)
			return lineset
		
		def load_axes():
			axes = o3d.geometry.LineSet()
			axes.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
			axes.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])          # X, Y, Z
			axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # R, G, B
			return axes
		
		vis = o3d.visualization.Visualizer()
		vis.create_window()
		# axis = load_axes()
		# vis.add_geometry(axis)

		view_control = vis.get_view_control()

		queue = mp.Queue()
		p = mp.Process(target=self.process_frames, args=(queue, ))
		p.start()
		keep_running = True
		all_cameras = o3d.geometry.LineSet()
		n = 0
		while keep_running:
			try:
				R, t = queue.get(block=False)
				if R is not None:
					#TODO:
					camera = create_camera()
					camera.transform(np.vstack((np.hstack((R, t)), [[0, 0, 0, 1]])))
					vis.add_geometry(camera)
					all_cameras.points.extend(camera.points)
					# view_control.set_lookat([0, 0, 0])
					view_control.set_front([-1, -1, 0])
					view_control.set_up([0, 0, -1])

					n += 1
					pass
			except: pass

			keep_running = keep_running and vis.poll_events()
		lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
		all_cameras.lines = o3d.utility.Vector2iVector([[lines[j][0]+i*5, lines[j][1]+i*5] for j in range(len(lines)) for i in range(n)])
		o3d.io.write_line_set('vo_lightglue.ply', all_cameras)
		vis.destroy_window()
		p.join()

	def process_frames(self, queue):
		
		extractor = SuperPoint(max_num_keypoints=500).eval().cuda()
		matcher = LightGlue(features='superpoint').eval().cuda() 
		
		def find_features(img):
			feats = extractor.extract(img)
			return feats
		
		def match_features(feats0, feats1):
			matches = matcher({'image0': feats0, 'image1': feats1})
			return matches

		def compute_pose(points0, points1, K):
			E, mask = cv.findEssentialMat(points0, points1, K, method=cv.RANSAC, prob=0.999, threshold=0.4)
			_, R, t, mask = cv.recoverPose(E, points0, points1, K, mask=mask)
			return R, t
		
		def triangulate(feats1b, feats2b, R, t, K, matches):
			pts1 = feats1b['keypoints'][matches[ : , 0]].cpu().numpy() 
			pts2 = feats2b['keypoints'][matches[ : , 0]].cpu().numpy() 
			pts1 = cv.undistortPoints(pts1, K, self.dist)
			pts2 = cv.undistortPoints(pts2, K, self.dist)
			pts4d = cv.triangulatePoints(np.eye(3, 4), np.hstack((R, t)), pts1[:, 0, :].T, pts2[:, 0, :].T)
			pts4d /= (pts4d[3] + 1e-8)
			return pts4d[:3].T
		
		def relScale(pts4d1, pts4d2):
			size = min(pts4d1.shape[0], pts4d2.shape[0])
			pts4d1 = pts4d1[:size]
			pts4d2 = pts4d2[:size]
			pts4d1_s = np.roll(pts4d1, axis=0, shift = 1) # calc two points distance
			pts4d2_s = np.roll(pts4d2, axis=0, shift = 1)
			scale = np.linalg.norm(pts4d2 - pts4d2_s, axis=1) / (np.linalg.norm(pts4d1 - pts4d1_s, axis=1) + 1e-8)
			scale = scale[~np.isnan(scale)]
			scale = scale[~np.isinf(scale)]
			return np.median(scale)

		R, t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
		queue.put((R, t))
		base_scale = 1
		img0 = load_image(self.frame_paths[0]).cuda()
		img1 = load_image(self.frame_paths[1]).cuda()
		feats0 = find_features(img0)
		feats1 = find_features(img1)
		matches01 = match_features(feats0, feats1)
		feats0b, feats1b, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
		matches01 = matches01['matches'].cpu().numpy()   
		points0 = feats0b['keypoints'][matches01[ : , 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
		points1 = feats1b['keypoints'][matches01[ : , 1]].cpu().numpy()  # coordinates in image #1, shape (K,2)
		R01, t01 = compute_pose(points0, points1, self.K)
		R = R @ R01
		t = t + np.dot(R, t01)
		queue.put((R, t))

		for frame_path in self.frame_paths[2:]:
			new_img_RGB = cv.imread(frame_path)
			img2 = load_image(frame_path).cuda()
			feats2 = find_features(img2)
			matches12 = match_features(feats1, feats2)
			feats2b, matches12 = [rbd(x) for x in [feats2, matches12]]
			matches12 = matches12['matches'].cpu().numpy()   
			points1 = feats1b['keypoints'][matches12[ : , 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
			points2 = feats2b['keypoints'][matches12[ : , 1]].cpu().numpy()  # coordinates in image #1, shape (K,2)
			R12, t12 = compute_pose(points1, points2, self.K)

			common_indices01 = np.where(np.in1d(matches01[:,1], matches12[:,0]))[0]
			common_matches01 = matches01[common_indices01, :]
			cloud01 = triangulate(feats0b, feats1b, R01, t01, self.K, common_matches01)
			common_indices12 = np.where(np.in1d(matches12[:,0], matches01[:,1]))[0]
			common_matches12 = matches12[common_indices12, :]			
			cloud12 = triangulate(feats1b, feats2b, R12, t12, self.K, common_matches12)
			
			scale = relScale(cloud01, cloud12)

			if scale == 0:
				continue
			base_scale *= np.cbrt(scale)
			t12 *= base_scale
			R = R @ R12
			t = t + np.dot(R, t12)
			queue.put((R, t))

			feats1 = feats2
			feats0b = feats1b.copy()
			feats1b = feats2b.copy()
			matches01 = matches12.copy()
			R01 = R12.copy()
			t01 = t12.copy()

			points2_list = points2.tolist()
			for pts in points2_list:
				new_img_RGB = cv.circle(new_img_RGB, (int(pts[0]), int(pts[1])), 3, (255, 0, 0), 1)
			cv.imshow('frame', new_img_RGB)
			if cv.waitKey(30) == 27: break
			
if __name__ == '__main__':
	torch.multiprocessing.set_start_method('spawn')
	parser = argparse.ArgumentParser()
	parser.add_argument('input', help='directory of sequential frames')
	parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
	args = parser.parse_args()

	vo = SimpleVO(args)
	vo.run()
