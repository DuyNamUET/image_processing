import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('digit.png', 0)
img_test = cv2.imread('num1.jpg', 0)

cell = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

x = np.array(cell)
# print(cell[49][99])
x_test = np.array(img_test)
x_test = cv2.resize(x_test,(16, 16))
cv2.bitwise_not(x_test, x_test, mask = x_test)
# print(x_test)
# cv2.imshow('', x_test)
# cv2.waitKey()

train = x[:,:].reshape(-1, 256).astype(np.float32)
test = x_test.reshape(-1, 256).astype(np.float32)

k = np.arange(10)
train_label = np.repeat(k, 500)[:, np.newaxis]

knn = cv2.ml.KNearest_create()
knn.train(train, 0, train_label)
result = knn.findNearest(test, 100)
print(result)