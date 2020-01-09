import os
import cv2

dirname = os.path.dirname(__file__)


class DIP:
    def reverse_rgb(self, img):
        b = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        r = img[:, :, 2].copy()

        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b

        return img


if __name__ == '__main__':
    img = cv2.imread(os.path.join(dirname, '../imori.jpg'))

    dip = DIP()
    img = dip.reverse_rgb(img)

    cv2.imwrite(os.path.join(dirname, 'imori_out.jpg'), img)
