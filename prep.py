from PIL import Image
import PIL.ImageOps as ImOps
import numpy as np
# import matplotlib.pyplot as plt

X = []
y = []

for i in range(1,16):
	for j in range(1,81):
		if j < 10:
			filename = "../gambar/resized/kertas-%d_0%d.jpg" % (i,j)
		else:
			filename = "../gambar/resized/kertas-%d_%d.jpg" % (i,j)
		img = Image.open(filename)
		inverted_image = ImOps.invert(img)
		matrix = np.array(inverted_image.convert('L'))
		label = j % 10
		if label == 0:
			label = 10
		label = label - 1

		X.append(matrix)
		y.append(label)


data = np.array(X)
label = np.array(y)

np.savez('aksara_sunda.npz', data=data, label=label)
# plt.subplot(111)
# plt.imshow(X[0], cmap=plt.get_cmap('gray'))
# plt.show()