import open3d as o3d
import numpy as np

all_cameras = o3d.io.read_line_set('vo_kitti.ply')
#print("all_cameras.points: ", np.asarray(all_cameras.points))
#print("all_cameras.lines: ", np.asarray(all_cameras.lines))

vis = o3d.visualization.Visualizer()
vis.create_window(width=960, height=540)
vis.add_geometry(all_cameras)
vis.run()
vis.destroy_window()