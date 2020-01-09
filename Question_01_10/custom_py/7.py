import os
import cv2
import numpy as np

dirname = os.path.dirname(__file__)


class DIP:
    def average_pool(self, img, pool_split=8):
        split_count_width = img.shape[0] // pool_split
        split_count_height = img.shape[1] // pool_split

        for i in range(pool_split):
            for j in range(pool_split):
                local = img[i*split_count_width:(i+1)*split_count_width,
                            j*split_count_height: (j+1)*split_count_height]

                self.__average_rgb(local)

        return img

    def __average_rgb(self, img):
        width, height, _ = img.shape

        img[:, :, 0] = np.mean(img[:, :, 0]).astype(np.int)
        img[:, :, 1] = np.mean(img[:, :, 1]).astype(np.int)
        img[:, :, 2] = np.mean(img[:, :, 2]).astype(np.int)


if __name__ == '__main__':

    img = cv2.imread(os.path.join(dirname, '../imori.jpg'))
    dip = DIP()
    out = dip.average_pool(img, pool_split=32)

    cv2.imwrite(os.path.join(dirname, 'imori_out7.jpg'), out)
