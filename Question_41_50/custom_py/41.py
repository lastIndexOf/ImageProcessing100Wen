import matplotlib.pyplot as plt
import numpy as np
import cv2

from math import cos, sin, pi, floor, ceil, sqrt
from os import path
import os

cwd = os.getcwd()
dirname = path.join(cwd, path.dirname(__file__))


class DIP:
    def canny(self, img):
        # 灰度
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 高斯模糊
        gauss = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=1.4, sigmaY=1.4)
        # Sobel滤波
        grad_x = cv2.Sobel(gauss, ddepth=-1, dx=1, dy=0,
                           ksize=3).astype(np.uint8)
        grad_y = cv2.Sobel(gauss, ddepth=-1, dx=0, dy=1,
                           ksize=3).astype(np.uint8)

        grad_x = np.maximum(grad_x, 1e-5)

        # edge = np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.uint8)
        edge = (np.abs(grad_x) + np.abs(grad_y)).astype(np.uint8)

        angle = np.arctan(grad_y / grad_x)
        angle = self.angle_quantization(angle)

        return grad_x, grad_y, edge, angle

    def angle_quantization(self, angle):
        angle = angle / np.pi * 180
        # angle[angle < -22.5] = 180 + angle[angle < -22.5]
        angle[angle < 0] = 180 + angle[angle < 0]
        # _angle = np.zeros_like(angle, dtype=np.uint8)
        # _angle[np.where(angle <= 22.5)] = 0
        # _angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
        # _angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
        # _angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

        return angle


if __name__ == '__main__':
    dip = DIP()
    img = cv2.imread(path.join(dirname, '../imori.jpg'))

    grad_x, grad_y, edge, angle = dip.canny(img)
    cv2.imwrite(path.join(dirname, './imori_out_edge_x41.jpg'), grad_x)
    cv2.imwrite(path.join(dirname, './imori_out_edge_y41.jpg'), grad_y)
    cv2.imwrite(path.join(dirname, './imori_out_edge41.jpg'), edge)
    cv2.imwrite(path.join(dirname, './imori_out_angle41.jpg'), angle)
