import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
import torch
# from lightglue import LightGlue, SuperPoint.
# from lightglue.utils import load_image, rbd
from kornia.feature import LoFTR
import kornia as K
import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32mb"

class SimpleVO:
	def __init__(self, args):
		camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
		self.K = camera_params['K']
		self.dist = camera_params['dist']
		
		self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))

	def run(self):
		vis = o3d.visualization.Visualizer()
		vis.create_window(width=960, height=540)
		
		queue = mp.Queue()
		p = mp.Process(target=self.process_frames, args=(queue, ))
		p.start()
		keep_running = True

		camera_frame_outline = np.array([[1, 2], [2, 3], [3, 4], [4, 1]])
		projected_line = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])

		optical_center = np.array([0, 0, 0]).reshape(3,1)
		camera_intrinsic_inv = np.linalg.inv(self.K)
		frame_point1 = camera_intrinsic_inv.dot(np.array([0, 0, 1])).reshape(3,1) 
		frame_point2 = camera_intrinsic_inv.dot(np.array([0, 359, 1])).reshape(3,1)
		frame_point3 = camera_intrinsic_inv.dot(np.array([639, 359, 1])).reshape(3,1)
		frame_point4 = camera_intrinsic_inv.dot(np.array([639, 0, 1])).reshape(3,1)

		while keep_running:
			try:
				R, t = queue.get(block=False)
				if R is not None:
					points = np.empty((0, 3), float)

					transform_center = R.dot(optical_center)+t
					transform_center = transform_center.reshape(1,3)
					points = np.append(points, transform_center, axis=0)
					transform_point1 = R.dot(frame_point1)+t
					transform_point1 = transform_point1.reshape(1,3)
					points = np.append(points, transform_point1, axis=0)
					transform_point2 = R.dot(frame_point2)+t
					transform_point2 = transform_point2.reshape(1,3)
					points = np.append(points, transform_point2, axis=0)
					transform_point3 = R.dot(frame_point3)+t
					transform_point3 = transform_point3.reshape(1,3)
					points = np.append(points, transform_point3, axis=0)
					transform_point4 = R.dot(frame_point4)+t
					transform_point4 = transform_point4.reshape(1,3)
					points = np.append(points, transform_point4, axis=0)

					cfo_line_set = o3d.geometry.LineSet()
					cfo_line_set.points = o3d.utility.Vector3dVector(points)
					cfo_line_set.lines = o3d.utility.Vector2iVector(camera_frame_outline)
					cfo_line_set.colors = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, 0], [0, 0, 0],[0, 1, 0]])

					pl_line_set = o3d.geometry.LineSet()
					pl_line_set.points = o3d.utility.Vector3dVector(points)
					pl_line_set.lines = o3d.utility.Vector2iVector(projected_line)
					pl_line_set.paint_uniform_color([1, 0, 0])

					vis.add_geometry(cfo_line_set)
					vis.add_geometry(pl_line_set)

					pass
								  
			except: pass
			
			keep_running = keep_running and vis.poll_events()
		vis.destroy_window()
		p.join()

	def process_frames(self, queue):

		def triangulate(feats1b, feats2b, R, t, K, matches):
			pts1 = feats1b[matches[ : ]].cpu().numpy()
			pts2 = feats2b[matches[ : ]].cpu().numpy() 
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
		
		def match_features(des1, des2):
			bf = cv.BFMatcher()
			matches = bf.knnMatch(des1, des2, k=2)
			good = []
			for m, n in matches:
				if m.distance < 0.7 * n.distance:
					good.append(m)
			return good
		
		def loftr_match_features(img1, img2):
			loftr = LoFTR('outdoor').eval().to('cuda')
			img1 = K.color.rgb_to_grayscale(img1).to('cuda')
			match = loftr({'image0': img1, 'image1': K.color.rgb_to_grayscale(img2).to('cuda')})
			kp1 = match['keypoints0'].cpu().numpy()
			kp2 = match['keypoints1'].cpu().numpy()
			confidence = match['confidence'].cpu().numpy()
			good = [i for i in range(len(kp1)) if confidence[i] > 0.8]
			return kp1, kp2, good
		
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
		
		def load_torch_image(fname):
			img = K.io.load_image(fname, K.io.ImageLoadType.RGB32)
			img = img[None]
			return img
		R, t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
		queue.put((R, t))
		base_scale = 1
		#Set feature detector 和 feature matcher
		# extractor = SuperPoint(max_num_keypoints=500).eval().cuda()  # load the extractor
		# matcher = LightGlue(features='superpoint').eval().cuda() 
		# Read first image and find feature.
		img0 = load_torch_image(self.frame_paths[0])
		img1 = load_torch_image(self.frame_paths[1])
		kp0, kp1, matches01 = loftr_match_features(img0, img1)
		E, mask = cv.findEssentialMat(kp1, kp0, cameraMatrix = self.K, method = cv.RANSAC, threshold = 0.5)
		# Find rotate matrix and translation matrix
		_, R01, t01, mask = cv.recoverPose(E, kp1, kp0, cameraMatrix = self.K, mask = mask)
		
		# feats1 = extractor.extract(img1)
		img0.cpu()

		for frame_path in self.frame_paths[2:]:
			# Read a new image and find feature.
			new_img_RGB = cv.imread(frame_path)
			img2 = load_torch_image(frame_path)
			# feats2 = extractor.extract(img2)
			kp2, kp3, matches12 = loftr_match_features(img1, img2)
			torch.cuda.empty_cache()

			# Find essential matrix
			E, mask = cv.findEssentialMat(kp3, kp2, cameraMatrix = self.K, method = cv.RANSAC, threshold = 0.5)

			# Find rotate matrix and translation matrix
			_, R12, t12, mask = cv.recoverPose(E, kp3, kp2, cameraMatrix = self.K, mask = mask)  

			# Calculate the consistent scale of t_k,k+1
			scale = 1
			if frame_path != self.frame_paths[1]:
				#matches01 = match_features(feats0, feats1)
				#cloud01 = triangulate_kp_is_coord(kp0, kp1, R01, t01, self.K, [m for m in matches01 if m.queryIdx in matches12])
				#cloud12 = triangulate_loftr(kp1, kp2, R12, t12, self.K, [m for m in matches12 if m in [k.trainIdx for k in matches01]])
				m01=[]
				m12=[]
				for i in matches01:
					for j in matches12:
						if np.linalg.norm(kp1[i]-kp2[j])<4:
							m01.append(i)
							m12.append(j)
				if(len(m01)):
					cloud01 = triangulate_loftr(kp0, kp1, R01, t01, self.K, m01)
					cloud12 = triangulate_loftr(kp2, kp3, R12, t12, self.K, m12)
					scale = relScale(cloud01, cloud12)
			print(scale, flush=True)

			# Calculate the pose of the current frame relative to the first frame
			base_scale *= np.cbrt(scale)         
			t12 *=  base_scale
			t = t + np.dot(R, t12)
			R = R @ R12
			#t = t + np.dot(R, t12)
			# Export R,t to queue
			queue.put((R, t))
			
			# Display feature points on the image
			points2_list = kp2
			for pts in points2_list:
				new_img_RGB = cv.circle(new_img_RGB, (int(pts[0]), int(pts[1])), 3, (255, 0, 0), 1)
			cv.imshow('Image with Feature Points', new_img_RGB)

			# Use the feature point of this frame for the next iteration
			# feats0 = feats1.copy()
			# feats1 = feats2.copy()
			kp0 = kp2.copy()
			kp1 = kp3.copy()
			matches01 = matches12.copy()
			R01 = R12.copy()
			t01 = t12.copy()
			
			img1.cpu()
			img2.cpu()

			if cv.waitKey(30) == 27: break

if __name__ == '__main__':
	print(torch.cuda.is_available())
	print(torch.cuda.device_count())
	torch.multiprocessing.set_start_method('spawn')
	parser = argparse.ArgumentParser()
	parser.add_argument('input', help='directory of sequential frames')
	parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
	args = parser.parse_args()

	vo = SimpleVO(args)
	vo.run()
