import matplotlib.pyplot as plt
import numpy as np
import cv2

from os import path
from math import cos, sin, pi, floor, ceil, sqrt
import os

cwd = os.getcwd()
dirname = path.join(cwd, path.dirname(__file__))


class DIP:
    def hpf(self, img):
        r = img[:, :, 2]
        g = img[:, :, 1]
        b = img[:, :, 0]

        out = img.copy()

        out[:, :, 2] = self.__hpf(r)
        out[:, :, 1] = self.__hpf(g)
        out[:, :, 0] = self.__hpf(b)

        return out

    def __hpf(self, img):
        H, W = img.shape

        dft = np.fft.fft2(img)
        dft_shift = np.fft.fftshift(dft)

        R_x = W // 2
        R_y = H // 2
        r = sqrt(R_x ** 2 + R_y ** 2) * 0.1

        x = np.arange(H).reshape(-1, 1).repeat(W, axis=1)
        y = np.tile(np.arange(W), (H, 1))

        dis = np.sqrt((x - R_x) ** 2 + (y - R_y) ** 2)

        mask = np.zeros_like(img)
        mask[dis > r] = 1.

        out = dft_shift * mask

        out = np.fft.ifftshift(out)
        out = np.abs(np.fft.ifft2(out))

        return out

    def gray(self, img):
        return 0.2126*img[:, :, 2]+0.7125*img[:, :, 1]+0.0722*img[:, :, 0]


if __name__ == '__main__':
    dip = DIP()
    img = cv2.imread(path.join(dirname, '../imori.jpg')).astype(np.float)

    out = dip.hpf(img)
    cv2.imwrite(path.join(dirname, './imori_out34.jpg'), out)
