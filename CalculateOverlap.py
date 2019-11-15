from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

object_to_gray = {
    'road':7,
    'sidewalk': 8,
    'building': 11,
    # 'fence': 13,
    # 'pole': 11,
    # 'traffic-light': 19,
    # 'traffic-sign': 20,
    # 'vegetation': 21,
    # 'terrain': 22,
    # 'sky': 23,
    # 'person':24,
    # 'rider': 25,
    # 'car': 26,
    # 'truck': 27,
    # 'bus': 28,
    # 'train':31,
    # 'motorbicycle':32,
    # 'bicycle': 33
}

object_to_rgb = {
    'road':(128, 64, 128),
    'sidewalk': (244, 35, 232),
    'building': (70, 70, 70),
    'fence': (190, 153, 153),
    'pole': (153, 153, 153),
    'traffic-light': (250, 170, 30),
    'traffic-sign': (220, 220, 0),
    'vegetation': (107, 42, 35),
    'terrain': (152, 251, 152),
    'sky': (70, 130, 180),
    'person': (220, 20, 60),
    'rider': (255, 0, 0),
    'car': (0, 0, 142),
    'truck': (0, 0, 70),
    'bus': (0, 60, 100),
    'train': (0, 80, 180),
    'motorbicycle': (0, 0, 230),
    'bicycle': (119, 11, 32)
}

def MahattanDistance(rgb_p, rgb_o):
    return abs(rgb_p[0] - rgb_o[0]) + abs(rgb_p[1] - rgb_o[1]) + abs(rgb_p[2] - rgb_o[2])

def AdjustImage(im):
    w, h = im.size
    s = int((w + h - abs(w - h)) / 2)
    l = int((w - s) / 2)
    u = int((h - s) / 2)
    im_new = im.crop((l, u, l+s, u+s))
    return im_new.resize((256, 256))

def ReturnGrayImage(path):
    img = Image.open(path)
    img = AdjustImage(img)
    gray_matrix = np.zeros((256, 256))
    # plt.imshow(img)
    # plt.show()

    # test_image = np.zeros((256, 256, 3))

    for i in range(256):
        for j in range(256):
            min_dis = 999
            min_obj = None
            for k in object_to_rgb.keys():
                temp_dis = MahattanDistance(img.getpixel((j, i)), object_to_rgb[k])
                if min_dis >= temp_dis:
                    min_dis = temp_dis
                    min_obj = k
            gray_matrix[i][j] = object_to_gray[min_obj]
            # test_image[j][i] = img.getpixel((i, j))

    plt.imshow(gray_matrix)
    plt.show()

    return gray_matrix

def SaveGrayImage(path, gray):
    plt.imshow(gray)
    plt.savefig(path)

# def UseThreshold(att, thresh):
#     for i in range(att):
#         for j in range(att[i]):
#             if att[i][j] >= thresh:
#                 att[i][j] = 1.
#             else:
#                 att[i][j] = 0.
#
#     return att
#
# def CalcOverlap(att, seg):
#     counter = {}
#
#     for k in object_to_gray.keys():
#         counter[object_to_gray[k]] = 0
#
#     for i in range(att):
#         for j in range(att[i]):
#             if att[i][j] == 1:
#                 counter[seg[i][j]] += 1
#
#     return counter

def CountOneOverlap(att, seg, thresh):
    counter = {}
    for k in object_to_gray.keys():
        counter[k] = ((att > thresh).float() * (seg == object_to_gray[k]).float()).sum()
        if counter[k] != 0:
            counter[k] /= (seg == object_to_gray[k]).float().sum()
    #     print(k)
    #     print(((att > thresh).float() * (seg == object_to_gray[k]).float()))
    # print(counter)
    return counter

def main():
    # SaveGrayImage('gray.jpg',ReturnGrayImage('roadimage.jpg'))
    att1 = [
        [1, 1, 1, 1],
        [1, 1, 0, 0.1],
        [1, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    img = [
        [7, 7, 7, 11],
        [7, 7, 7, 11],
        [11, 11, 11, 11],
        [11, 11, 11, 11]
    ]
    att1 = torch.Tensor(att1)
    img = torch.Tensor(img)
    # print(att1)
    # print(img)
    CountOneOverlap(att1, img, 0.5)

if __name__ == '__main__':
    main()

    # (self.att_A>self.opt.thresh).float()