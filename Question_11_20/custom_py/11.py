import os
import cv2
import numpy as np

dirname = os.path.dirname(__file__)


class DIP:
    def mean_filter(self, img):
        '''均值滤波'''
        if len(img.shape) == 3:
            W, H, C = img.shape
        else:
            img = np.expand_dims(img, axis=-1)
            W, H, C = img.shape

        out = np.zeros((W+2, H+2, C))
        for w in range(1, W+1):
            for h in range(1, H+1):
                for c in range(C):
                    out[w, h, c] = np.mean(
                        img[w-1:w+2, h-1:h+2, c]).astype(np.uint8)

        return out[1:W+1, 1:H+1, :]


if __name__ == '__main__':

    img = cv2.imread(os.path.join(dirname, '../imori.jpg'))
    dip = DIP()
    out = dip.mean_filter(img)

    cv2.imwrite(os.path.join(dirname, 'imori_out11.jpg'), out)
