import cv2
import numpy as np

a = np.array([[4.25725, -4.600382], [5.85725, 8.599619], [-3.0927505, 0.59961843], [-0.6427505, -1.5003817] \
    , [2.6072495, 9.049618], [5.2072496, 4.0496182]])
a *= 100
b = np.array([[373, 248], [1673, 254], [826, 971], [639, 707], [1685, 527], [1217, 258]])


h, status = cv2.findHomography(b, a)
print(h)
c = np.array([[1036, 527, 1]])

new = np.dot(h, c.T)
print(new)


np.save("homography.npy", h)