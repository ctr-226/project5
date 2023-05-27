# -*-coding = utf-8 -*-
"""
1. Image_flip:翻转图片
2. Image_traslation:平移图片
3. Image_rotate:旋转图片
4. Image_noise:添加噪声
"""
import glob
import os
import random
from random import choice

import cv2
import numpy as np

'''
此文件并未实际使用，可以忽略
'''

def Image_flip(img, mode):
    """
    :param img:原始图片矩阵
    :return: 0-垂直； 1-水平； -1-垂直&水平
    """
    if img is None:
        return
    if mode == 2:
        return img
    img_new = cv2.flip(img, mode)
    return img_new


def Image_traslation(img):
    """
    :param img: 原始图片矩阵
    :return: [1, 0, 100]-宽右移100像素； [0, 1, 100]-高下移100像素
    """
    traslation = 20

    paras_wide = [[1, 0, traslation], [1, 0, -traslation]]
    paras_height = [[0, 1, traslation], [0, 1, -traslation]]
    rows, cols = img.shape[:2]
    img_shift = np.float32([choice(paras_wide), choice(paras_height)])
    border_value = tuple(int(x) for x in choice(choice(img)))
    img_new = cv2.warpAffine(img, img_shift, (cols, rows), borderValue=border_value)
    return img_new


def Image_rotate(img):
    """
    :param img:原始图片矩阵
    :return:旋转中心，旋转角度，缩放比例
    """
    rows, cols = img.shape[:2]
    rotate_core = (cols / 2, rows / 2)
    rotate_angle = [60, -60, 45, -45, 90, -90, 210, 240, -210, -240]
    paras = cv2.getRotationMatrix2D(rotate_core, choice(rotate_angle), 1)
    border_value = tuple(int(x) for x in choice(choice(img)))
    img_new = cv2.warpAffine(img, paras, (cols, rows), borderValue=border_value)
    return img_new


def Image_noise(img, noise):
    """
    :param img:原始图片矩阵
    :return: 0-高斯噪声，1-椒盐噪声
    """
    paras = [0, 0]
    gaussian_class = choice(paras)
    noise_ratio = [0.05, 0.06, 0.08]
    if gaussian_class == 1:
        output = np.zeros(img.shape, np.uint8)
        prob = choice(noise_ratio)
        thres = 1 - prob
        # print('prob', prob)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = img[i][j]
        return output
    else:
        mean = 0
        var = noise
        # print('var', var)
        img = np.array(img / 255, dtype=float)
        # noise = np.random.normal(mean, var ** 0.5, img.shape)
        noise = np.random.normal(mean, var ** 0.5, (img.shape[0], img.shape[1]))
        # noise = np.array([noise])
        # noise = noise.transpose(1, 2, 0)
        out = img + noise
        if out.min() < 0:
            low_clip = -1
        else:
            low_clip = 0
        out = np.clip(out, 0.0, 1.0)
        out = np.uint8(out * 255)
        return out


def Image_Expansion(_path_read, _path_write, _enhance_num):
    path_read = _path_read
    path_write = _path_write
    enhance_num = _enhance_num  # 扩增数目
    image_list = [x for x in os.listdir(path_read)]
    existed_img = len(image_list)
    while enhance_num > 0:
        img = choice(image_list)
        image = cv2.imread(os.path.join(path_read, img), cv2.IMREAD_GRAYSCALE)  # 灰度图读入
        algorithm = [1, 2, 3, 4]
        suffix = ['flip', 'translation', 'rotation', 'noise']
        noises = [0.001, 0.002, 0.003, 0.005]
        modes = [-1, 0, 1, 2]
        random_process = choice(algorithm)
        if random_process == 1:
            image = Image_flip(image, mode=choice(modes))
        # elif random_process == 2:
        #     image = Image_traslation(image)
        # elif random_process == 3:
        #     image = Image_rotate(image)
        else:
            image = Image_flip(image, mode=choice(modes))  # 旋转后再添加噪声
            image = Image_noise(image, noise=choice(noises))
        image_dir = os.path.join(path_write, img.split('.')[0] + '_' + suffix[random_process - 1] + '_' +
                                 str(enhance_num + existed_img - 1).zfill(6) + '.tif')
        cv2.imwrite(image_dir, image)
        enhance_num -= 1


def ImageExpansion(_path_read, _path_write):
    path_read = _path_read
    path_write = _path_write
    image_list = [x for x in os.listdir(path_read)]
    existed_img = len(image_list)
    for img in image_list:
        image = cv2.imread(os.path.join(path_read, img), cv2.IMREAD_GRAYSCALE)
        suffix = ['flip', 'noise']
        for mode in [2]:  # [-1, 0, 1, 2]
            im = Image_flip(image, mode)
            image_dir = os.path.join(path_write, img.split('.')[0] + '_' + suffix[0] + '_' +
                                     str(mode) + '.tif')
            cv2.imwrite(image_dir, im)
        for mode in [2]:
            for noise in [0.005]:
                im = Image_flip(image, mode)
                im = Image_noise(im, noise)
                image_dir = os.path.join(path_write, img.split('.')[0] + '_' + suffix[0] + '_' +
                                         str(mode) + suffix[1] + '_' + str(noise) + '.tif')
                cv2.imwrite(image_dir, im)


def Image_ExpansionTest(_path_read, _path_write, _enhance_num):
    path_read = _path_read
    path_write = _path_write
    enhance_num = _enhance_num
    image_list = [x for x in os.listdir(path_read)]
    existed_img = len(image_list)
    while enhance_num > 0:
        img = choice(image_list)
        image = cv2.imread(path_read + img, cv2.IMREAD_COLOR)
        algorithm = [1, 2, 3, 4]
        # random_process = choice(algorithm)
        random_process = 4
        if random_process == 1:
            print(np.array(image).shape)
            image = Image_flip(image)
            print(np.array(image).shape)
        elif random_process == 2:
            image = Image_traslation(image)
        elif random_process == 3:
            image = Image_rotate(image)
        else:
            image = Image_noise(image)
        image_dir = path_write + str(enhance_num + existed_img - 1).zfill(5) + '.tif'
        # cv2.imwrite(image_dir, image)
        enhance_num -= 1


if __name__ == "__main__":
    image_path = "../dataset/cifar10/train"
    # 遍历训练集中的类别文件夹
    train_class_folders = glob.glob(os.path.join(image_path, "*"))
    classname = os.listdir(image_path)
    for image_class in classname:
        _path_read = os.path.join(image_path, image_class)
        _path_write = os.path.join(image_path + '_A', image_class)
        try:
            os.makedirs(_path_write, 0o0755)  # mkdir一次只能创建一级目录， makedirs可创建多级目录
        except:
            pass
        image_list = [x for x in os.listdir(_path_read)]
        existed_img = len(image_list)
        print(image_class, existed_img)
        enhance_num = existed_img * 6
        Image_Expansion(_path_read, _path_write, enhance_num)

        # ImageExpansion(_path_read=_path_read, _path_write=_path_write)
