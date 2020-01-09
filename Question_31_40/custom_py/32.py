import matplotlib.pyplot as plt
import numpy as np
import cv2

from os import path
from math import cos, sin, pi, floor, ceil
import os

cwd = os.getcwd()
dirname = path.join(cwd, path.dirname(__file__))


class DIP:
    def fourier_transform(self, img):
        gray_img = self.gray(img)

        dft = np.fft.fft2(gray_img)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * np.log(np.abs(dft_shift))

        idft_shift = np.fft.ifftshift(dft_shift)
        idft = np.fft.ifft2(idft_shift)
        img_ = np.abs(idft)

        return magnitude_spectrum, img_

    def gray(self, img):
        return 0.2126*img[:, :, 2]+0.7125*img[:, :, 1]+0.0722*img[:, :, 0]


if __name__ == '__main__':
    dip = DIP()
    img = cv2.imread(path.join(dirname, '../imori.jpg')).astype(np.float)

    magnitude_spectrum, out = dip.fourier_transform(img)
    cv2.imwrite(
        path.join(dirname, './imori_magnitude_spectrum32.jpg'), magnitude_spectrum)
    cv2.imwrite(path.join(dirname, './imori_out32.jpg'), out)
