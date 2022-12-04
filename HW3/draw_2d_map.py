import numpy as np

import matplotlib.pyplot as plt
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
pcd = pcd.select_by_index(np.where(points[:,1] < -0.002)[0]) 
#o3d.visualization.draw_geometries([pcd])

color = pcd.colors
points = np.asarray(pcd.points)
x_points = points[:, 0] * 1000
y_points = points[:, 1] * 1000
z_points = points[:, 2] * 1000


temp = np.array([x_points, z_points])
theta = np.radians(277)
c, s = np.cos(theta), np.sin(theta)
R = np.array(((c, -s), (s, c)))

temp = np.dot(R, temp)
temp[1, :] = -temp[1, :]

temp[0, :] -= np.min(temp[0])
temp[1, :] -= np.min(temp[1])

x_min = np.min(temp[0])
x_max = np.max(temp[0])
y_min = np.min(temp[1])
y_max = np.max(temp[1])

print(x_min, x_max, y_min, y_max)

plt.figure(figsize=(25, 15), dpi=80)
plt.scatter(temp[0], temp[1], s = 20, c = color)
plt.axis('off')
plt.savefig('test.png')
plt.show()


