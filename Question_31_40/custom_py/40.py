import matplotlib.pyplot as plt
import numpy as np
import cv2

from math import cos, sin, pi, floor, ceil, sqrt
from os import path
import os

cwd = os.getcwd()
dirname = path.join(cwd, path.dirname(__file__))


class DIP:
    # Y 色域做量化
    Q1 = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
                   (12, 12, 14, 19, 26, 58, 60, 55),
                   (14, 13, 16, 24, 40, 57, 69, 56),
                   (14, 17, 22, 29, 51, 87, 80, 62),
                   (18, 22, 37, 56, 68, 109, 103, 77),
                   (24, 35, 55, 64, 81, 104, 113, 92),
                   (49, 64, 78, 87, 103, 121, 120, 101),
                   (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)

    # Cb Cr 做量化
    Q2 = np.array(((17, 18, 24, 47, 99, 99, 99, 99),
                   (18, 21, 26, 66, 99, 99, 99, 99),
                   (24, 26, 56, 99, 99, 99, 99, 99),
                   (47, 66, 99, 99, 99, 99, 99, 99),
                   (99, 99, 99, 99, 99, 99, 99, 99),
                   (99, 99, 99, 99, 99, 99, 99, 99),
                   (99, 99, 99, 99, 99, 99, 99, 99),
                   (99, 99, 99, 99, 99, 99, 99, 99)), dtype=np.float32)

    def RGB2YCBCR(self, img):
        H, W, C = img.shape

        ycbcr = np.zeros((H, W, C), dtype=np.float32)

        ycbcr[:, :, 0] = 0.2990 * img[:, :, 2] + \
            0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 0]
        ycbcr[:, :, 1] = -0.1687 * img[:, :, 2] - 0.3313 * \
            img[:, :, 1] + 0.5 * img[:, :, 0] + 128.
        ycbcr[:, :, 2] = 0.5 * img[:, :, 2] - 0.4187 * \
            img[:, :, 1] - 0.0813 * img[:, :, 0] + 128.

        return ycbcr

    def YCBCR2RGB(self, img):
        H, W, C = ycbcr.shape

        out = np.zeros((H, W, C), dtype=np.float32)
        out[:, :, 2] = ycbcr[:, :, 0] + (ycbcr[:, :, 2] - 128.) * 1.4020
        out[:, :, 1] = ycbcr[:, :, 0] - \
            (ycbcr[:, :, 1] - 128.) * 0.3441 - (ycbcr[:, :, 2] - 128.) * 0.7139
        out[:, :, 0] = ycbcr[:, :, 0] + (ycbcr[:, :, 1] - 128.) * 1.7718

        return out

    def dct(self, img):
        out = np.zeros(img.shape)

        out[:, :, 2] = self.__dct(img[:, :, 2])
        out[:, :, 1] = self.__dct(img[:, :, 1])
        out[:, :, 0] = self.__dct(img[:, :, 0])

        return out

    def idct(self, img):
        out = np.zeros(img.shape)

        out[:, :, 2] = self.__idct(img[:, :, 2], key='Y')
        out[:, :, 1] = self.__idct(img[:, :, 1], key='Cb')
        out[:, :, 0] = self.__idct(img[:, :, 0], key='Cr')

        return out

    def __dct(self, img):
        H, W = img.shape
        T = 8

        out = np.zeros((H, W))
        for x in range(0, H, T):
            for y in range(0, W, T):
                out[x:x+T, y:y+T] = cv2.dct(np.float32(img[x:x+T, y:y+T]))

        return out

    def __idct(self, img, key):
        H, W = img.shape
        T = 8

        out = np.zeros((H, W))
        for x in range(0, H, T):
            for y in range(0, W, T):
                out[x:x+T, y:y +
                    T] = cv2.idct(self.__quantization(img[x:x+T, y:y+T], key))

        out = np.clip(out, 0, 255)
        out = np.round(out).astype(np.uint8)

        return out

    def __quantization(self, img, key):
        if key == 'Y':
            return np.round(img / DIP.Q1) * DIP.Q1
        else:
            return np.round(img / DIP.Q2) * DIP.Q2


if __name__ == '__main__':
    dip = DIP()
    img = cv2.imread(path.join(dirname, '../imori.jpg')).astype(np.float)

    ycbcr = dip.RGB2YCBCR(img)
    ycbcr_dct = dip.dct(ycbcr)
    ycbcr_idct = dip.idct(ycbcr_dct)
    out = dip.YCBCR2RGB(ycbcr_idct)

    img2 = dip.dct(img)
    out2 = dip.idct(img2)
    cv2.imwrite(path.join(dirname, './imori_out1_40.jpg'), out)
    cv2.imwrite(path.join(dirname, './imori_out2_40.jpg'), out2)
