import matplotlib.pyplot as plt
import numpy as np
import cv2

from os import path
import os

cwd = os.getcwd()
dirname = path.join(cwd, path.dirname(__file__))


class DIP:
    def hist_equal(self, img):
        H, W, C = img.shape
        S = H * W * C

        out = img.copy()
        sum_ = 0.
        for i in range(1, 255):
            ind = np.where(img == i)
            sum_ += len(img[ind])
            current = sum_ * 255 / S
            out[ind] = current

        return out


if __name__ == '__main__':
    dip = DIP()
    img = cv2.imread(path.join(dirname, '../imori.jpg')).astype(np.float)

    out = dip.hist_equal(img)
    plt.hist(np.ravel(out), bins=255, rwidth=0.8, range=(0, 255))
    plt.savefig(path.join(dirname, './imori_out_hist23.jpg'))
    cv2.imwrite(path.join(dirname, './imori_out23.jpg'), out)
