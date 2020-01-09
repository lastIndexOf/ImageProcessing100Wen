import os
import cv2

dirname = os.path.dirname(__file__)


class DIP:
    def decrise_color(self, img, n_color=64):
        img = img // n_color * n_color + n_color / 2

        return img


if __name__ == '__main__':

    img = cv2.imread(os.path.join(dirname, '../imori.jpg'))
    dip = DIP()
    out = dip.decrise_color(img)

    cv2.imwrite(os.path.join(dirname, 'imori_out6.jpg'), out)
