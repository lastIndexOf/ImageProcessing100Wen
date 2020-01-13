import matplotlib.pyplot as plt
import numpy as np
import cv2

from math import cos, sin, pi, floor, ceil, sqrt
from os import path
import os

cwd = os.getcwd()
dirname = path.join(cwd, path.dirname(__file__))


class DIP:
    def dct(self, img):
        r = img[:, :, 2]
        g = img[:, :, 1]
        b = img[:, :, 0]

        out = img.copy()

        out[:, :, 2] = self.__dct(r)
        out[:, :, 1] = self.__dct(g)
        out[:, :, 0] = self.__dct(b)

        return out

    def idct(self, img):
        r = img[:, :, 2]
        g = img[:, :, 1]
        b = img[:, :, 0]

        out = img.copy()

        out[:, :, 2] = self.__idct(r)
        out[:, :, 1] = self.__idct(g)
        out[:, :, 0] = self.__idct(b)

        return out

    def __dct(self, img):
        return cv2.dct(np.float32(img))

    def __idct(self, img):
        return cv2.idct(img)

    def gray(self, img):
        return 0.2126*img[:, :, 2]+0.7125*img[:, :, 1]+0.0722*img[:, :, 0]


if __name__ == '__main__':
    dip = DIP()
    img = cv2.imread(path.join(dirname, '../imori.jpg')).astype(np.float)

    dct_img = dip.dct(img)
    idct_img = dip.idct(dct_img)
    cv2.imwrite(path.join(dirname, './imori_out37.jpg'), idct_img)
