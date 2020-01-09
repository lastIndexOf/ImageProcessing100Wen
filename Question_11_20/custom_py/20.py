import matplotlib.pyplot as plt
import numpy as np
import cv2

from os import path
import os

cwd = os.getcwd()
dirname = path.join(cwd, path.dirname(__file__))


class DIP:
    def hist(self, img):
        plt.hist(np.ravel(img), bins=255, rwidth=0.8, range=(0, 255))
        plt.savefig(path.join(dirname, './imori_dark_out20.jpg'))
        plt.show()


if __name__ == '__main__':
    dip = DIP()
    img = cv2.imread(path.join(dirname, '../imori_dark.jpg')).astype(np.float)

    dip.hist(img)
