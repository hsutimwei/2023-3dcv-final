import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
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

        #Set feature detector 和 feature matcher
        extractor = SuperPoint(max_num_keypoints=500).eval().cuda()  # load the extractor
        matcher = LightGlue(features='superpoint').eval().cuda() 

        # Read first image and find feature.
        img1 = load_image(self.frame_paths[0]).cuda()
        
        feats1 = extractor.extract(img1)

        for frame_path in self.frame_paths[1:]:
            # Read a new image and find feature.
            new_img_RGB = cv.imread(frame_path)
            img2 = load_image(frame_path).cuda()
            feats2 = extractor.extract(img2)
            # Do match.
            matches12 = matcher({'image0': feats1, 'image1': feats2})
            feats1_rm, feats2_rm, matches12 = [rbd(x) for x in [feats1, feats2, matches12]]  # remove batch dimension

            matches12 = matches12['matches'].cpu().numpy()   # indices with shape (K,2)
            points1 = feats1_rm['keypoints'][matches12[ : , 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
            points2 = feats2_rm['keypoints'][matches12[ : , 1]].cpu().numpy()  # coordinates in image #1, shape (K,2)

            # Find essential matrix
            E, mask = cv.findEssentialMat(points2, points1, cameraMatrix = self.K, method = cv.RANSAC, threshold = 0.5)

            # Find rotate matrix and translation matrix
            _, R12, t12, mask = cv.recoverPose(E, points2, points1, cameraMatrix = self.K, mask = mask)  

            # Calculate the consistent scale of t_k,k+1
            if frame_path != self.frame_paths[1]:
                common_indices01 = np.where(np.in1d(matches01[:,1], matches12[:,0]))[0]
                common_matches01 = matches01[common_indices01, :]
                cloud01 = triangulate(feats0_rm, feats1_rm, R01, t01, self.K, common_matches01)
                common_indices12 = np.where(np.in1d(matches12[:,0], matches01[:,1]))[0]
                common_matches12 = matches12[common_indices12, :]			
                cloud12 = triangulate(feats1_rm, feats2_rm, R12, t12, self.K, common_matches12)
                scale = relScale(cloud01, cloud12)
            else:
                scale = 1

            # Calculate the pose of the current frame relative to the first frame
            base_scale *= np.cbrt(scale)         
            t12 *=  base_scale
            t = t + np.dot(R, t12)
            R = R @ R12
            #t = t + np.dot(R, t12)

            # Export R,t to queue
            queue.put((R, t))
            
            # Display feature points on the image
            points1_list = points1.tolist()
            for pts in points1_list:
                img2 = cv.circle(new_img_RGB, (int(pts[0]), int(pts[1])), 3, (255, 0, 0), 1)
            cv.imshow('Image with Feature Points', img2)

            # Use the feature point of this frame for the next iteration
            feats1 = feats2
            feats0_rm = feats1_rm.copy()
            feats1_rm = feats2_rm.copy()
            matches01 = matches12.copy()
            R01 = R12.copy()
            t01 = t12.copy()

            if cv.waitKey(30) == 27: break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
