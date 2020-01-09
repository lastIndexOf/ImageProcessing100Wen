import matplotlib.pyplot as plt
import numpy as np
import cv2

from os import path
from math import cos, sin, pi, floor, ceil
import os

cwd = os.getcwd()
dirname = path.join(cwd, path.dirname(__file__))


class DIP:
    def affine_transform(self, img, x_scale=1., y_scale=1., rotate=0, x_translate=0, y_translate=0):
        H, W, C = img.shape

        E = np.array([
            [x_scale * cos(rotate), -sin(rotate), x_translate],
            [sin(rotate), y_scale * cos(rotate), y_translate],
            [0, 0, 1]]
        )

        out = np.zeros((H, W, C))

        for x in range(H):
            for y in range(W):
                point = E.dot(np.array([x, y, 1]))
                x_, y_ = point[0], point[1]

                if x_ > 0 and x_ < H - 1 and y_ > 0 and y_ < W - 1:
                    out[floor(x_), floor(y_)] = img[x, y]
                    out[floor(x_), ceil(y_)] = img[x, y]
                    out[ceil(x_), floor(y_)] = img[x, y]
                    out[ceil(x_), ceil(y_)] = img[x, y]

        return out


if __name__ == '__main__':
    dip = DIP()
    img = cv2.imread(path.join(dirname, '../imori.jpg')).astype(np.float)

    out = dip.affine_transform(
        img, x_translate=64, y_translate=-26, rotate=(45/180*pi))
    cv2.imwrite(path.join(dirname, './imori_out28.jpg'), out)
