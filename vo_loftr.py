import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
from kornia.feature import LoFTR
import kornia as K

import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32mb"

class SimpleVO:
	def __init__(self, args):
		camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
		self.K = camera_params['K']
		self.dist = camera_params['dist']

		self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))

	def run(self):
		RATIO = 10
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
		vis.create_window(width=960, height=540)
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
		o3d.io.write_line_set('vo_loftr.ply', all_cameras)
		vis.destroy_window()
		p.join()

	def process_frames(self, queue):

		def find_features(img):
			sift = cv.SIFT_create()
			kp, des = sift.detectAndCompute(img, None)
			return kp, des
		
		def match_features(des1, des2):
			bf = cv.BFMatcher()
			matches = bf.knnMatch(des1, des2, k=2)
			good = []
			for m, n in matches:
				if m.distance < 0.7 * n.distance:
					good.append(m)
			return good
		
		def load_torch_image(fname):
			img = K.io.load_image(fname, K.io.ImageLoadType.RGB32)
			img = img[None]
			return img

		def loftr_match_features(img1, img2):
			loftr = LoFTR('outdoor').eval().to('cuda')
			with torch.no_grad():
				match = loftr({'image0': K.color.rgb_to_grayscale(img1).to('cuda'), 'image1': K.color.rgb_to_grayscale(img2).to('cuda')})
			kp1 = match['keypoints0'].cpu().numpy()
			kp2 = match['keypoints1'].cpu().numpy()
			confidence = match['confidence'].cpu().numpy()
			good = [i for i in range(len(kp1)) if confidence[i] > 0.7]
			feat1 = match['feat0'][good].cpu().numpy()
			feat2 = match['feat1'][good].cpu().numpy()

			return kp1, kp2, good, feat1, feat2

		def compute_pose(kp1, kp2, matches, K):
			pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
			pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
			E, mask = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC, prob=0.999, threshold=0.4)
			_, R, t, mask = cv.recoverPose(E, pts1, pts2, K, mask=mask)
			return R, t
		
		def compute_pose_loftr(kp1, kp2, matches, K):
			pts1 = np.array([kp1[i] for i in matches])
			pts2 = np.array([kp2[i] for i in matches])
			E, mask = cv.findEssentialMat(pts2, pts1, K, method=cv.RANSAC, prob=0.999, threshold=0.4)
			_, R, t, mask = cv.recoverPose(E, pts2, pts1, K, mask=mask)
			return R, t
		
		def triangulate(kp1, kp2, R, t, K, matches):
			pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
			pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
			pts1 = cv.undistortPoints(pts1, K, self.dist)
			pts2 = cv.undistortPoints(pts2, K, self.dist)
			pts4d = cv.triangulatePoints(np.eye(3, 4), np.hstack((R, t)), pts1[:, 0, :].T, pts2[:, 0, :].T)
			pts4d /= (pts4d[3] + 1e-8)
			return pts4d[:3].T
		
		def triangulate_kp_is_coord(kp1, kp2, R, t, K, matches):
			pts1 = np.array([kp1[m.queryIdx] for m in matches])
			pts2 = np.array([kp2[m.trainIdx] for m in matches])
			pts1 = cv.undistortPoints(pts1, K, self.dist)
			pts2 = cv.undistortPoints(pts2, K, self.dist)
			pts4d = cv.triangulatePoints(np.eye(3, 4), np.hstack((R, t)), pts1[:, 0, :].T, pts2[:, 0, :].T)
			pts4d /= (pts4d[3] + 1e-8)
			return pts4d[:3].T

		def triangulate_loftr(kp1, kp2, R, t, K, matches):
			pts1 = np.array([kp1[m] for m in matches])
			pts2 = np.array([kp2[m] for m in matches])
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

		img0 = load_torch_image(self.frame_paths[0])
		img1 = load_torch_image(self.frame_paths[1])
		kp0, kp1, matches01, des0, des1 = loftr_match_features(img0, img1)

		R01, t01 = compute_pose_loftr(kp0, kp1, matches01, self.K)
		R = R @ R01
		t = t + np.dot(R, t01)
		queue.put((R, t))

		result = cv.VideoWriter('frames.avi', cv.VideoWriter_fourcc(*'MJPG'), 10, (640, 360))

		for frame_path in self.frame_paths[2:]:
			img2 = load_torch_image(frame_path)
			torch.cuda.empty_cache()
			kp1, kp2, matches12, des1, des2 = loftr_match_features(img1, img2)

			torch.cuda.empty_cache()
			R12, t12 = compute_pose_loftr(kp1, kp2, matches12, self.K)
			matches01 = match_features(des0, des1)

			cloud01 = triangulate_kp_is_coord(kp0, kp1, R01, t01, self.K, [m for m in matches01 if m.queryIdx in matches12])
			cloud12 = triangulate_loftr(kp1, kp2, R12, t12, self.K, [m for m in matches12 if m in [k.trainIdx for k in matches01]])
			
			scale = relScale(cloud01, cloud12)
			if scale == 0:
				continue
			base_scale *= np.cbrt(scale)
			t12 *= base_scale
			R = R @ R12
			t = t + np.dot(R, t12)
			queue.put((R, t))
			img1 = img2
			kp0 = kp1.copy()
			kp1 = kp2.copy()
			des0 = des1.copy()
			des1 = des2.copy()
			# matches01 = matches12.copy()
			R01 = R12.copy()
			t01 = t12.copy()
			img2 = cv.imread(frame_path)
			for pts in [kp2[m] for m in matches12]:
				img2 = cv.circle(img2, (int(pts[0]), int(pts[1])), 1, (255, 0, 0), 1)
			cv.imshow('frame', img2)
			result.write(img2)
			if cv.waitKey(30) == 27: break
		result.release()
			
if __name__ == '__main__':
	
	torch.multiprocessing.set_start_method('spawn')
	parser = argparse.ArgumentParser()
	parser.add_argument('input', help='directory of sequential frames')
	parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
	args = parser.parse_args()

	vo = SimpleVO(args)
	vo.run()
