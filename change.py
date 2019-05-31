import sys
import cv2
import numpy as np

def change(img, a, b):
    new_img = np.asarray(a * img + b)
    new_img[new_img > 255] = 255
    new_img[new_img < 0] = 0
    return new_img

def main():
    alpha = int(sys.argv[1])
    beta = int(sys.argv[2])

    img = cv2.imread("festo.jpg")
    new_img = change(img, alpha, beta)

    cv2.imshow("", new_img)
    cv2.waitKey()

if __name__ == "__main__":
    main()