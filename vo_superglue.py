import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
import torch
import sys
from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)
from time import sleep
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
                    n+=1
                    pass
            except: pass

            keep_running = keep_running and vis.poll_events()
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        all_cameras.lines = o3d.utility.Vector2iVector([[lines[j][0]+i*5, lines[j][1]+i*5] for j in range(len(lines)) for i in range(n)])
        o3d.io.write_line_set('vo_superglue.ply', all_cameras)
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

        def compute_pose(kp1, kp2, K):

            E, mask = cv.findEssentialMat(kp1, kp2, K, method=cv.RANSAC, prob=0.999, threshold=0.4)
            _, R, t, mask = cv.recoverPose(E, kp1, kp2, K, mask=mask)
            return R, t
        
        def triangulate(mkpts1, mkpts2, R, t, K):
            mkpts1 = cv.undistortPoints(np.array(mkpts1), K, self.dist)
            mkpts2 = cv.undistortPoints(np.array(mkpts2), K, self.dist)
            pts4d = cv.triangulatePoints(np.eye(3, 4), np.hstack((R, t)), mkpts1[:, 0, :].T, mkpts2[:, 0, :].T)
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

        def drawmatch(image0,image1,mkpts0,mkpts1):
            H0, W0 = image0.shape
            H1, W1 = image1.shape
            H, W = max(H0, H1), W0 + W1 + 10

            out = 255*np.ones((H, W), np.uint8)
            out[:H0, :W0] = image0
            out[:H1, W0+10:] = image1
            out = np.stack([out]*3, -1)
            mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)

            for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
                cv.line(out, (x0, y0), (x1 + 10 + W0, y1),
                color=(0,255,0), thickness=1, lineType=cv.LINE_AA)
            # display line end-points as circles
                cv.circle(out, (x0, y0), 2, (0,255,0), -1, lineType=cv.LINE_AA)
                cv.circle(out, (x1 + 10 + W0, y1), 2, (0,255,0), -1,
                   lineType=cv.LINE_AA)
            cv.imshow("superglue", out)
            cv.waitKey(10000)
   
   
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': -1
            },
            'superglue': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        keys = ['keypoints', 'scores', 'descriptors']
        matching = Matching(config).eval().to(device)
        R, t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        queue.put((R, t))
        base_scale = 1
        img0 = cv.imread(self.frame_paths[0],cv.IMREAD_GRAYSCALE)
        img1 = cv.imread(self.frame_paths[1],cv.IMREAD_GRAYSCALE)
        frame_tensor0 = frame2tensor(img0, device)
        frame_tensor1 = frame2tensor(img1, device)
        with torch.no_grad():
            data0 = matching.superpoint({'image': frame_tensor0})
        data0 = {k+'0': data0[k] for k in keys}
        data0['image0'] = frame_tensor0
        with torch.no_grad():
            data1 = matching.superpoint({'image': frame_tensor1})
        data1_1 = {k+'1': data1[k] for k in keys}
        data1_1['image1'] = frame_tensor1
        with torch.no_grad():
            pred = matching({**data0, **data1_1})#superpoint 跟frame_tensor
        
        kpts0 = data0['keypoints0'][0].cpu().numpy()
        
        kpts1 = data1_1['keypoints1'][0].cpu().numpy()
        
        matches01 = pred['matches0'][0].cpu().numpy()
        
        data1 = {k+'0': data1[k] for k in keys}
        data1['image0'] = frame_tensor1
        
        preframe=data1
        valid01 = matches01 > -1
        mkpts0 = kpts0[valid01]
        mkpts1 = kpts1[matches01[valid01]]

        #drawmatch(img0,img1,mkpts0[[1,100,200,300,400,500]],mkpts1[[1,100,200,300,400,500]])
        
        R01, t01 = compute_pose(mkpts0, mkpts1, self.K)
        R = R @ R01
        t = t + np.dot(R, t01)
        
        queue.put((R, t))
        
        i=0
        for frame_path in self.frame_paths[2:]:
            torch.cuda.empty_cache()
            i+=1
            img = cv.imread(frame_path)
            img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            frame_tensor2 = frame2tensor(img2, device)
            with torch.no_grad():
                data2 = matching.superpoint({'image': frame_tensor2})
            data2_2 = {k+'1': data2[k] for k in keys}
            data2_2['image1'] = frame_tensor2
            with torch.no_grad():
                pred2 = matching({**preframe, **data2_2})
            
            kpts1 = preframe['keypoints0'][0].cpu().numpy()

            
            kpts2 = data2_2['keypoints1'][0].cpu().numpy()
            
            matches12 = pred2['matches0'][0].cpu().numpy()
            valid12 = matches12 > -1
            
            mkpts1_1 = kpts1[valid12]
            mkpts2 = kpts2[matches12[valid12]]
            with torch.no_grad():
                preframe = matching.superpoint({'image': frame_tensor2})
            preframe = {k+'0': preframe[k] for k in keys}
            preframe['image0'] = frame_tensor2

            R12, t12 = compute_pose(mkpts1_1, mkpts2, self.K)
            index01=[]
            index12=[]
            for i,point1 in enumerate(mkpts1):
                for j,point2 in enumerate(mkpts1_1):
                    if point1[0]==point2[0] and point1[1]==point2[1]:
                        index01.append(i)
                        index12.append(j)

            mkpts0_0=mkpts0[index01]
            mkpts1_0=mkpts1[index01]
            #mkpts1_1_0=mkpts1_1[index12]
            mkpts2_0=mkpts2[index12]

            
            cloud01 = triangulate(mkpts0_0, mkpts1_0, R01, t01, self.K)
            cloud12 = triangulate(mkpts1_0, mkpts2_0, R12, t12, self.K)
            scale = relScale(cloud01, cloud12)
            if scale == 0:
                continue
            base_scale *= np.cbrt(scale)
            t12 *= base_scale
            R = R @ R12
            t = t + np.dot(R, t12)
            queue.put((R, t))
            img1 = img2.copy()
            kpts0 = kpts1
            kpts1 = kpts2
            mkpts0 = mkpts1_1
            mkpts1 = mkpts2
            matches01 = matches12.copy()
            
            R01 = R12.copy()
            t01 = t12.copy()
            for pts in mkpts2:
                img = cv.circle(img, (int(pts[0]), int(pts[1])), 1, (255, 0, 0), 1)
            cv.imshow('frame', img)
            if cv.waitKey(30) == 27: break
            
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
