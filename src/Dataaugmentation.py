import glob
import os

from PIL import Image
from torchvision import transforms

# 设置输入目录
train_input_dir = "../dataset/cifar10/train"

# 定义数据增强转换
data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
])

# 遍历训练集中的类别文件夹
train_class_folders = glob.glob(os.path.join(train_input_dir, "*"))

for class_folder in train_class_folders:
    # 遍历类别文件夹中的图像文件
    image_files = glob.glob(os.path.join(class_folder, "*.png"))

    for image_file in image_files:
        # 加载图像
        image = Image.open(image_file)

        # 应用数据增强转换
        augmented_image = data_augmentation(image)

        # 获取文件名和扩展名
        file_name, ext = os.path.splitext(os.path.basename(image_file))

        # 构建输出文件路径
        output_file = os.path.join(class_folder, f"a{file_name}{ext}")

        # 保存增强后的图像
        augmented_image.save(output_file)

        print(f"Processed train image: {image_file}")

print("Data augmentation completed.")
