import matplotlib.pyplot as plt
import numpy as np
import cv2

from os import path
from math import floor
import os

cwd = os.getcwd()
dirname = path.join(cwd, path.dirname(__file__))


class DIP:
    def nn_insert(self, img, gain_x=1.5, gain_y=1.5):
        H, W, C = img.shape

        gain_h = round(H*gain_x)
        gain_w = round(W*gain_y)

        h_ind = np.floor(np.arange(gain_h) / gain_x).astype(np.int)
        w_ind = np.floor(np.arange(gain_w) / gain_y).astype(np.int)

        out = img[:, w_ind][h_ind, :]

        return out


if __name__ == '__main__':
    dip = DIP()
    img = cv2.imread(path.join(dirname, '../imori.jpg')).astype(np.float)

    out = dip.nn_insert(img, gain_x=2, gain_y=2)
    cv2.imwrite(path.join(dirname, './imori_out25.jpg'), out)
