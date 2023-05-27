import json

import matplotlib.pyplot as plt

# 读取JSON文件
with open('../history/CNN1.json', 'r') as file:
    data = json.load(file)

# 提取训练损失、测试损失、训练准确率和测试准确率数据
train_loss = data['train_loss']
test_loss = data['test_loss']
train_accuracy = data['train_accuracy']
test_accuracy = data['test_accuracy']

# 绘制训练损失和测试损失曲线
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.savefig('../history/loss_plot.png')  # 保存损失曲线图
plt.close()

# 绘制训练准确率和测试准确率曲线
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(test_accuracy, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Testing Accuracy')
plt.legend()
plt.savefig('../history/accuracy_plot.png')  # 保存准确率曲线图
plt.close()
