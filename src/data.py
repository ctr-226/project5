import pickle
import glob
import cv2
import numpy as np
import os

label_name = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def split_dict(dict, save_path):
    for im_idx, im_data in enumerate(dict[b"data"]):
        im_name = dict[b"filenames"][im_idx]
        im_label = dict[b"labels"][im_idx]
        print(im_name, im_label, im_data)

        im_lable_name = label_name[im_label]  # label序号转化为具体名称
        im_data = np.reshape(im_data, [3, 32, 32])
        im_data = np.transpose(im_data, [1, 2, 0])  # 对空间矩阵的进行转置
        # cv2.imshow("data",im_data)
        # cv2.waitKey(0)
        if not os.path.exists("{}/{}".format(save_path, im_lable_name)):
            os.mkdir("{}/{}".format(save_path, im_lable_name))
        cv2.imwrite("{}/{}/{}".format(save_path, im_lable_name,
                                      im_name.decode("utf-8")), im_data)


def main():
    dataset_root = '../dataset'
    # file_path = 'D:\AI Math Theory\project5\dataset\cifar-10-batches-py\data_batch_1'
    train_batch = glob.glob("../dataset/cifar-10-batches-py/data*")
    test_batch = glob.glob("../dataset/cifar-10-batches-py/test_batch")
    for batch in train_batch:
        save_path = '../dataset/cifar10/train'
        dict = unpickle(batch)
        split_dict(dict, save_path)
    for batch in test_batch:
        save_path = '../dataset/cifar10/test'
        dict = unpickle(batch)
        split_dict(dict, save_path)


if __name__ == '__main__':
    main()
