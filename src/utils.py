import json
import os

import matplotlib.pyplot as plt


def read_dataset(root: str, is_train: bool):
    # 遍历文件夹，一个文件夹对应一个类别
    classes = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    classes.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(classes))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('./classes.json', 'w') as json_file:
        json_file.write(json_str)

    images_path = []  # 存储训练集的所有图片路径
    images_label = []  # 存储训练集图片对应索引信息

    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG", ".tif", ".jpeg"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in classes:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        for img_path in images:
            images_path.append(img_path)
            images_label.append(image_class)
    if is_train:
        print("{} images for training.".format(len(images_path)))
    else:
        print("{} images for testing.".format(len(images_path)))

    plot_image = False
    if plot_image:
        plt.figure(figsize=(8, 6), dpi=300)
        # 绘制每种类别个数柱状图
        plt.bar(range(len(classes)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(classes)), classes)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('Training Set Class Distribution')
        plt.show()

    return images_path, images_label


def load_config(path):
    with open(path, mode='r', encoding='utf-8') as f:
        return json.load(f)
