import cv2
import numpy
import matplotlib.pyplot as plt

img = cv2.imread('digit.png', 0)

cells = [numpy.hsplit(row, 100) for row in numpy.vsplit(img, 50)]

x = numpy.array(cells)
# print(x.size)
# xx = x[0, 0].reshape(-1, 256)

train = x[:,:50].reshape(-1, 256).astype(numpy.float32)
test = x[:,50:100].reshape(-1, 256).astype(numpy.float32)

k = numpy.arange(10)
train_label = numpy.repeat(k, 250)[:, numpy.newaxis]

knn = cv2.ml.KNearest_create()
knn.train(train, 0, train_label)
result = knn.findNearest(test, 5)
print(result[2])


# print(train_label)
# cv2.imshow('',xx)
# cv2.waitKey()
