import matplotlib.pyplot as plt
import numpy as np
import cv2

from os import path
import os

cwd = os.getcwd()
dirname = path.join(cwd, path.dirname(__file__))


class DIP:
    def hist_mani(self, img, loc=128, scale=52):
        mean, std = np.mean(img), np.std(img)

        out = img.copy()
        # (out - mean) / std == 转为均值为0，标准差为1
        out = (out - mean) / std * scale + loc

        np.clip(out, 0, 255)

        return out


if __name__ == '__main__':
    dip = DIP()
    img = cv2.imread(path.join(dirname, '../imori_dark.jpg')).astype(np.float)

    out = dip.hist_mani(img)
    plt.hist(np.ravel(out), bins=255, rwidth=0.8, range=(0, 255))
    plt.savefig(path.join(dirname, './imori_dark_out_hist22.jpg'))
    cv2.imwrite(path.join(dirname, './imori_dark_out22.jpg'), out)
