import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('frank.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(np.array(img).shape)
ans = np.reshape(img, [1,48*48])[0]

for _ in range(len(ans)):
	print(ans[_], end=" ")

plt.imshow(np.resize(ans, [48, 48]), cmap='gray')
plt.show()
