#  $$ Y = 0.2126\ R + 0.7152\ G + 0.0722\ B $$
import os
import cv2
import numpy as np


dirname = os.path.dirname(__file__)


class DIP:
    def grap_img(self, img):
        b = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        r = img[:, :, 2].copy()

        out = 0.2126 * r + 0.7125 * g + 0.0722 * b
        out.astype(np.uint8)

        return out


if __name__ == '__main__':
    img = cv2.imread(os.path.join(dirname, '../imori.jpg'))
    dip = DIP()
    out = dip.grap_img(img)

    cv2.imwrite(os.path.join(dirname, 'imori_out2.jpg'), out)
