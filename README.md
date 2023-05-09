主要结构说明：

config目录存储训练时的各种参数： 

    model_name:指定VGG结构，现在咱们是自定义CNN，所以没什么用了
    dataset:不用修改
    num_classes：分类数量，也不用改
    channels：图像通道数，RGB图片3通道
    features：卷积层特征提取器的结构，数字代表卷积层通道数，R表示RELU激活函数，M表示最大池化层
    classifier_in：特征提取器输出的向量长度，也是分类器输入的维度，需要自己根据网络结构计算一下
    classifier：线性层分类器的结构，数字表示隐藏层单元数，最后一个是10表示一共10类
    其它参数应该比较好理解

dataset目录存储cifar10数据集：

    dataset/cifar-10-batches-py：此目录是cifar10原始数据文件
    dataset/cifar10：提取为图片文件后的目录，运行src/data.py后生成

src目录存储各种源代码：

    data.py：提取数据集，第一次应该先运行这个文件
    Dataset.py：pytorch的dataset
    model.py：定义CNN结构的代码
    train.py：训练网络
    utils.py：各类工具函数

跑程序的顺序：

先运行data.py生成一次数据集，然后在config/目录下写好自己的设置文件
运行python train.py --config_name=--config_name=config/***.json指定具体的配置文件
    