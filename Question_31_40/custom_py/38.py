import matplotlib.pyplot as plt
import numpy as np
import cv2

from math import cos, sin, pi, floor, ceil, sqrt
from os import path
import os

cwd = os.getcwd()
dirname = path.join(cwd, path.dirname(__file__))


class DIP:
    Q = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
                  (12, 12, 14, 19, 26, 58, 60, 55),
                  (14, 13, 16, 24, 40, 57, 69, 56),
                  (14, 17, 22, 29, 51, 87, 80, 62),
                  (18, 22, 37, 56, 68, 109, 103, 77),
                  (24, 35, 55, 64, 81, 104, 113, 92),
                  (49, 64, 78, 87, 103, 121, 120, 101),
                  (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)

    def dct(self, img, T=8):
        r = img[:, :, 2]
        g = img[:, :, 1]
        b = img[:, :, 0]

        out = img.copy()

        out[:, :, 2] = self.__dct(r, T)
        out[:, :, 1] = self.__dct(g, T)
        out[:, :, 0] = self.__dct(b, T)

        return out

    def idct(self, img, K=8):
        r = img[:, :, 2]
        g = img[:, :, 1]
        b = img[:, :, 0]

        out = img.copy()

        out[:, :, 2] = self.__idct(r, K)
        out[:, :, 1] = self.__idct(g, K)
        out[:, :, 0] = self.__idct(b, K)

        return out

    def gray(self, img):
        return 0.2126*img[:, :, 2]+0.7125*img[:, :, 1]+0.0722*img[:, :, 0]

    def __dct(self, img, T=8):
        # 8*8
        H, W = img.shape

        out = np.zeros((H, W))
        for x in range(0, H, T):
            for y in range(0, W, T):
                out[x:x+T, y:y+T] = cv2.dct(np.float32(img[x:x+T, y:y+T]))

        return out[:H, :W]

    def __idct(self, img, K=8):
        H, W = img.shape

        out = np.zeros((H, W))
        for x in range(0, H, K):
            for y in range(0, W, K):
                out[x:x+K, y:y +
                    K] = cv2.idct(self.__quantization(img[x:x+K, y:y+K]))

        out = np.clip(out, 0, 255)
        out = np.round(out).astype(np.uint8)

        return out[:H, :W]

    def __quantization(self, img):
        return np.round(img / DIP.Q) * DIP.Q


if __name__ == '__main__':
    dip = DIP()
    img = cv2.imread(path.join(dirname, '../imori.jpg')).astype(np.float)

    dct_img = dip.dct(img)
    idct_img = dip.idct(dct_img)
    cv2.imwrite(path.join(dirname, './imori_out38.jpg'), idct_img)
