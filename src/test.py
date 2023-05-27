import os
from multiprocessing import Process, Semaphore

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from absl import app
from absl import flags
# from tqdm import tqdm
from absl import logging
from torchvision import transforms
from tqdm import tqdm

import model
from Dataset import MyDataSet
from utils import read_dataset, load_config

experiments_path = '..'  # 项目根目录
flags.DEFINE_string('config_name', 'config/VGG11.json', help='')

def load_cifar10(path, is_train, name, batch_size):
    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
        ])}
    mode = 'train' if is_train else 'test'
    data_folder = os.path.join('..', path, name, mode)
    images_path, images_label = read_dataset(data_folder, is_train=is_train)
    dataset = MyDataSet(images_path=images_path,
                        images_class=images_label,
                        transform=data_transform[mode])
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=is_train,
                                              num_workers=4,
                                              pin_memory=True,
                                              collate_fn=dataset.collate_fn)

    return data_loader, images_path


@torch.no_grad()
def validation(net, val_set, criterion, device):
    num_data = 0
    corrects = 0
    test_loss = []

    # Test loop
    net.eval()
    _softmax = torch.nn.Softmax(dim=1)
    for i, data in enumerate(tqdm(val_set)):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        predictions = net(images).to(device)
        loss = criterion(predictions, labels)
        test_loss.append(loss.cpu())
        predictions = _softmax(predictions)

        _, predictions = torch.max(predictions.data, 1)
        labels = labels.type(torch.LongTensor)
        num_data += labels.size(0)
        corrects += (predictions == labels.to(device)).sum().item()

    accuracy = 100 * corrects / num_data
    return accuracy, np.mean(test_loss)


def get_weight(folder):
    _, _, files = next(os.walk(folder))
    for file in files:
        ext_name = os.path.splitext(file)[-1]
        if ext_name == '.pth':
            return os.path.join(folder, file)
    return None


def factorize(n):
    # 如果n是负数或0，返回None
    if n <= 0:
        return None
    # 如果n是1，返回(1, 1)
    if n == 1:
        return (1, 1)
    # 初始化两个因数为n的平方根的向下取整和向上取整
    a = int(n ** 0.5)
    b = a + 1
    # 循环，直到找到满足条件的因数或者a为1
    while a > 1:
        # 如果n能被a整除，返回(a, n // a)
        if n % a == 0:
            return (a, n // a)
        # 否则，将a减一，将b加一
        else:
            a -= 1
            b += 1
    # 如果循环结束，没有找到满足条件的因数，返回(1, n)
    return (1, n)


def plot_feature_map(feature_map, channels, save_path):
    a, b = factorize(channels)  # 根据通道数确定画图格式
    plt.figure(figsize=(16, 16))  # 画布大小
    for channel in range(channels):
        ax = plt.subplot(a, b, channel + 1)
        plt.imshow(feature_map[channel, :, :].cpu())  # 灰度图参数cmap="gray"
    plt.savefig(save_path, dpi=300)
    plt.close()


def run(dataset, dropout_rate, model_name, features, classifier, classifier_in, device, experiments_path):
    valid_set, _ = load_cifar10('dataset', False, dataset, 1)
    net = model.cnn(features, classifier, classifier_in, dropout_rate).to(device)
    # 加载模型文件
    model_path = os.path.join(experiments_path, f'model/{model_name}')
    weight_path = get_weight(folder=model_path)
    if weight_path is not None:
        net.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        raise FileExistsError
    save_path = os.path.join(experiments_path, f'experiment/{model_name}')

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    # 生成存储文件夹
    for i in range(10):
        try:
            os.makedirs(os.path.join(save_path, str(i)), exist_ok=True)
        except OSError:
            pass

    # Visualize feature maps
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # 存储卷积层权重参数
    model_weights = []
    # 存储卷积层
    conv_layers = []
    # 获取模型结构列表
    model_children = list(net.children())
    # 卷积层数量计数
    counter = 0
    # 遍历提取卷积层和权重参数
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
            model_children[i].register_forward_hook(get_activation(i))
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                if type(model_children[i][j]) == nn.Conv2d:
                    counter += 1
                    model_weights.append(model_children[i][j].weight)
                    conv_layers.append(model_children[i][j])
                    model_children[i][j].register_forward_hook(get_activation(j))
    print(f"Total convolution layers: {counter}")
    print(conv_layers)
    # 读取测试集图片的文件名
    images_list = []
    for i, path in enumerate(valid_set.dataset.images_path):
        images_list.append(os.path.basename(valid_set.dataset.images_path[i]).split('.')[0])

    net.eval()
    _softmax = torch.nn.Softmax(dim=1)
    for i, data in enumerate(tqdm(valid_set)):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        predictions = net(images).to(device)
        # 绘制特征图
        process = []
        for key in activation.keys():
            feature_map = activation[key].squeeze(0)
            channels = feature_map.shape[0]
            fig_path = f"{save_path}/{str(labels.item())}/{images_list[i]}-feature-{key}.jpg"
            process.append(Process(target=plot_feature_map, args=(feature_map, channels, fig_path)))
        # a, b = factorize(channels)  # 根据通道数确定画图格式
        # plt.figure(figsize=(16, 16))  # 画布大小
        # for channel in range(channels):
        # 	ax = plt.subplot(a, b, channel + 1)
        # 	plt.imshow(feature_map[channel, :, :].cpu())  # 灰度图参数cmap="gray"
        # plt.savefig(fig_path, dpi=300)
        # plt.close()
        for p in process:
            p.start()
        for p in process:
            p.join()

    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数

    test_accuracy, test_loss = validation(net, valid_set, criterion, device)
    # 输出测试的信息
    logging.info(
        f'test_loss={test_loss:.4f} | test_accuracy={test_accuracy:.2f}'
    )


def main(_):
    FLAGS = flags.FLAGS
    config_name = FLAGS.config_name
    config = load_config(os.path.join(experiments_path, config_name))
    dataset = config['dataset']
    classes = config['num_classes']
    channels = config['channels']
    features = config['features']
    classifier = config['classifier']
    classifier_in = config['classifier_in']
    epochs = config['epochs']
    batch_size = config['batch_size']
    device = config['device']
    momentum = config['momentum']
    lr = config['lr']
    lr_step = config['lr_step']
    lr_decay = config['lr_decay']

    weight_decay = config['weight_decay']
    dropout_rate = config['dropout_rate']

    model_name = config['model_name']

    run(dataset, dropout_rate, model_name, features=features, classifier=classifier,
        classifier_in=classifier_in, device=device, experiments_path=experiments_path)


if __name__ == '__main__':
    app.run(main)
