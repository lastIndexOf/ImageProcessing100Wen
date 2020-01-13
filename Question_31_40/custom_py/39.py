import matplotlib.pyplot as plt
import numpy as np
import cv2

from math import cos, sin, pi, floor, ceil, sqrt
from os import path
import os

cwd = os.getcwd()
dirname = path.join(cwd, path.dirname(__file__))


class DIP:
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

        out = np.clip(out, 0, 255)
        out = out.astype(np.uint8)

        return out


if __name__ == '__main__':
    dip = DIP()
    img = cv2.imread(path.join(dirname, '../imori.jpg')).astype(np.float)

    ycbcr = dip.RGB2YCBCR(img)
    ycbcr[:, :, 0] = ycbcr[:, :, 0] * .7
    out = dip.YCBCR2RGB(ycbcr)
    cv2.imwrite(path.join(dirname, './imori_out39.jpg'), out)
