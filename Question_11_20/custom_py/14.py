import os
import cv2
import numpy as np

dirname = os.path.dirname(__file__)


class DIP:
    def gray_img(self, img):
        '''Gray = R*0.299 + G*0.587 + B*0.114'''
        return (0.229*img[:, :, 2]+0.587*img[:, :, 1]+0.114*img[:, :, 0]).astype(np.uint8)

    def differential_filter(self, img, k_size=3):
        H, W = img.shape

        pad = k_size // 2
        Kv = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float)
        Kh = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float)

        out = np.zeros((H+2*pad, W+2*pad))
        out[pad: H+pad, pad: W+pad] = img.copy()

        temp = out.copy()

        out_h = out.copy()
        out_v = out.copy()

        for h in range(H):
            for w in range(W):
                out_h[pad+h, pad+w] = np.sum(temp[h:h+k_size, w:w+k_size]*Kh)
                out_v[pad+h, pad+w] = np.sum(temp[h:h+k_size, w:w+k_size]*Kv)

        out_h = np.clip(out_h, 0, 255)
        out_v = np.clip(out_v, 0, 255)

        out_h = out_h[pad:H+pad, pad:W+pad]
        out_v = out_v[pad:H+pad, pad:W+pad]

        return out_v, out_h


if __name__ == '__main__':

    img = cv2.imread(os.path.join(dirname, '../imori.jpg'))
    dip = DIP()
    out_v, out_h = dip.differential_filter(dip.gray_img(img))

    cv2.imwrite(os.path.join(dirname, 'imori_out14_v.jpg'), out_v)
    cv2.imwrite(os.path.join(dirname, 'imori_out14_h.jpg'), out_h)
    cv2.imwrite(os.path.join(dirname, 'imori_out14.jpg'), out_h+out_v)
