import numpy as np

import open3d as o3d

color = np.load("color01.npy")
points = np.load("point.npy")
points = o3d.utility.Vector3dVector(points)

pcd = o3d.geometry.PointCloud(points)
pcd.colors = o3d.utility.Vector3dVector(color)

print(pcd.get_max_bound())
print(pcd.get_min_bound())

points = np.asarray(pcd.points)
pcd = pcd.select_by_index(np.where(points[:,1] > -0.035)[0]) 
points = np.asarray(pcd.points)
pcd = pcd.select_by_index(np.where(points[:,1] <  0.010)[0]) 
o3d.visualization.draw_geometries([pcd])



