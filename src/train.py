import argparse
import datetime
from absl import flags
from absl import app
import os
import time
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
# from tqdm import tqdm
from absl import logging
from torchvision import transforms
from Dataset import MyDataSet
from utils import read_dataset, load_config
import json
import model
from tqdm import tqdm

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

def train(net, train_set, device, optimizer, criterion):
    _loss = []  # 记录每轮训练的train_loss
    num_data = 0
    corrects = 0
    net.train()
    for i, data in enumerate(tqdm(train_set)):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        predictions = net(images).to(device)  # 模型预测值
        loss = criterion(predictions, labels)
        _loss.append(loss.item())  # 每个batch训练的loss
        _, predictions = torch.max(predictions.data, 1)
        num_data += labels.size(0)
        corrects += (predictions == labels.to(device)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_accuracy = 100 * corrects / num_data
    if net.lr_scheduler:
        lr = net.lr_scheduler.get_last_lr()[0]
        net.lr_scheduler.step()

    return train_accuracy, np.mean(_loss)


def run(epochs, dataset, classes, channels, batch_size,
        lr, lr_step, lr_decay, weight_decay, momentum, dropout_rate,
        model_name, device, experiments_path):

    train_set, _ = load_cifar10('dataset', True, dataset, batch_size)
    valid_set, _ = load_cifar10('dataset', False, dataset, batch_size)
    lr_scheduler = None
    net = model.vgg(
        model_name='vgg11', num_classes=classes, lr_scheduler=lr_scheduler
    ).to(device)

    save_path = os.path.join(experiments_path, f'model/{model_name}')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    history_path = os.path.join(experiments_path, 'history')
    if not os.path.exists(history_path):
        os.makedirs(history_path, exist_ok=True)

    history = {
        'train_loss': [],
        'test_loss': [],
        'train_accuracy': [],
        'test_accuracy': [],
    }

    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )  # SGD随机梯度下降优化器
    for epoch in range(epochs):
        train_accuracy, train_loss = train(net, train_set, device, optimizer, criterion)

        test_accuracy, test_loss = validation(net, valid_set, criterion, device)
        # 输出每轮训练的信息
        logging.info(
            f'Epoch: {epoch + 1:03d}/{epochs:03d} | train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | lr={lr} | train_accuracy={train_accuracy:.2f} | test_accuracy={test_accuracy:.2f}'
        )

        history['train_loss'].append(float(train_loss)) # 记录每轮的train_loss
        history['test_loss'].append(float(test_loss))  # 记录每轮的test_loss
        history['train_accuracy'].append(float(train_accuracy))  # 记录每轮的test_accuracy
        history['test_accuracy'].append(float(test_accuracy))  # 记录每轮的test_accuracy

        # 存储每一轮的模型
        if experiments_path:
            torch.save(net.state_dict(), os.path.join(save_path, f'model-{epoch + 1:03d}.pth'))
        with open(os.path.join(history_path, f'{model_name}.json'), mode='w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=True, indent=2)

def main(_):
    FLAGS = flags.FLAGS
    config_name = FLAGS.config_name
    config = load_config(os.path.join(experiments_path, config_name))
    dataset = config['dataset']
    classes = config['num_classes']
    channels = config['channels']
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

    run(epochs, dataset, classes, channels, batch_size,
        lr, lr_step, lr_decay, weight_decay, momentum, dropout_rate,
        model_name, device, experiments_path)


if __name__ == '__main__':
    app.run(main)