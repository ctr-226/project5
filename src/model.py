import torch
import torch.nn as nn

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=False, lr_scheduler=None):
        super(VGG, self).__init__()
        self.lr_scheduler = lr_scheduler
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 1024),  # 尺寸根据输入进行计算调整
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)  # 最后一层无需relu，直接输出
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Net(nn.Module):
    def __init__(self, features, classifier, num_classes=10, init_weights=False, lr_scheduler=None):
        super(Net, self).__init__()
        self.lr_scheduler = lr_scheduler
        self.features = features
        self.classifier = classifier
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.features():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.classifier():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "R":
            layers += [nn.ReLU(True)]  # 将卷积层中的激活函数换成线性函数 如f(x)=x
        elif v == "T":
            layers += [nn.Tanh()]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d]
            in_channels = v
    return nn.Sequential(*layers)


def make_classifier(cfg: list, classifier_in, dropout_rate=0.5):
    layers = []
    in_channels = classifier_in
    for i, v in enumerate(cfg):
        if v == "R":
            layers += [nn.ReLU(True)]
        elif v == "D":
            layers += [nn.Dropout(p=dropout_rate)]
        elif v == "T":
            layers += [nn.Tanh()]
        else:
            linear = nn.Linear(in_channels, v)
            layers += [linear]  # 将卷积层中的激活函数换成线性函数 如f(x)=x
            in_channels = v
    return nn.Sequential(*layers)


# 配置文件，对应不同的VGG结构 M-maxpool
cfgs = {
    'vgg11': [64, 'R', 'M', 128, 'R', 'M', 256, 'R', 256, 'R', 'M', 512, 'R', 512, 'R', 'M', 512, 'R', 512, 'R', 'M'],
    'vgg13': [64, 'R', 64, 'R', 'M', 128, 'R', 128, 'R', 'M', 256, 'R', 256, 'R', 'M', 512, 'R', 512, 'R', 'M', 512,
              'R', 512, 'R', 'M'],
    'vgg16': [64, 'R', 64, 'R', 'M', 128, 'R', 128, 'R', 'M', 256, 'R', 256, 'R', 256, 'R', 'M', 512, 'R', 512, 'R',
              512, 'R', 'M', 512, 'R', 512, 'R', 512, 'R', 'M'],
    'vgg19': [64, 'R', 64, 'R', 'M', 128, 'R', 128, 'R', 'M', 256, 'R', 256, 'R', 256, 'R', 256, 'R', 'M', 512, 'R',
              512, 'R', 512, 'R', 512, 'R', 'M', 512, 'R', 512, 'R', 512, 'R', 512, 'R', 'M'],
}


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]
    model = VGG(make_features(cfg), **kwargs)
    return model


def cnn(features, classifier, classifier_in, dropout_rate):
    model = Net(make_features(features), make_classifier(classifier, classifier_in, dropout_rate=dropout_rate))
    return model
