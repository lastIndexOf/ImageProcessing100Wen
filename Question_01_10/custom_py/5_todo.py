import os
import cv2

dirname = os.path.dirname(__file__)


class DIP:
    def thresholding(self, img):

        b = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        r = img[:, :, 2].copy()

        out = 0.2126 * r + 0.7125 * g + 0.0722 * b

        out[out < 128] = 0
        out[out > 128] = 255

        return out


if __name__ == '__main__':

    img = cv2.imread(os.path.join(dirname, '../imori.jpg'))
    dip = DIP()
    out = dip.thresholding(img)

    cv2.imwrite(os.path.join(dirname, 'imori_out5.jpg'), out)
