import matplotlib.pyplot as plt
import numpy as np
import cv2

from os import path
import os

cwd = os.getcwd()
dirname = path.join(cwd, path.dirname(__file__))


class DIP:
    def hist_normalize(self, img, w_range=(0, 255)):
        '''直方图归一化'''
        min_, max_ = np.min(img), np.max(img)
        w_min, w_max = w_range

        out = img.copy()

        out = (out - min_) * (w_max - w_min) / (max_ - min_) + w_min
        np.clip(out, 0, 255)

        return out


if __name__ == '__main__':
    dip = DIP()
    img = cv2.imread(path.join(dirname, '../imori_dark.jpg')).astype(np.float)

    out = dip.hist_normalize(img)
    plt.hist(np.ravel(out), bins=255, rwidth=0.8, range=(0, 255))
    plt.savefig(path.join(dirname, './imori_dark_out_hist21.jpg'))
    cv2.imwrite(path.join(dirname, './imori_dark_out21.jpg'), out)
