import matplotlib.pyplot as plt
import numpy as np
import cv2

from os import path
import os

cwd = os.getcwd()
dirname = path.join(cwd, path.dirname(__file__))


class DIP:
    def gamma_correct(self, img):
        out = img.copy()

        gamma = 1 / 2.2
        out = (out / 255) ** gamma * 255

        return out


if __name__ == '__main__':
    dip = DIP()
    img = cv2.imread(path.join(dirname, '../imori_gamma.jpg')).astype(np.float)

    out = dip.gamma_correct(img)
    plt.hist(np.ravel(out), bins=255, rwidth=0.8, range=(0, 255))
    cv2.imwrite(path.join(dirname, './imori_gamma_out24.jpg'), out)
