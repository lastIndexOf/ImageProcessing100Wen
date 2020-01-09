import os
import cv2
import numpy as np

dirname = os.path.dirname(__file__)


class DIP:
    def max_min_filter(self, img):
        b = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        r = img[:, :, 2].copy()

        img = np.expand_dims((0.2126 * r + 0.7125 * g + 0.0722 * b), axis=-1)

        W, H, C = img.shape

        out = np.zeros((W+2, H+2, C))
        for w in range(1, W+1):
            for h in range(1, H+1):
                for c in range(C):
                    temp = img[w-1:w+2, h-1:h+2, c]
                    out[w, h, c] = (np.max(temp) - np.min(temp)
                                    ).astype(np.uint8)

        return out[1:W+1, 1:H+1, 0]


if __name__ == '__main__':

    img = cv2.imread(os.path.join(dirname, '../imori.jpg'))
    dip = DIP()
    out = dip.max_min_filter(img)

    cv2.imwrite(os.path.join(dirname, 'imori_out13.jpg'), out)
