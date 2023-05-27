import os
import random

from shutil import copyfile

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


def main():
    dataset_path = '../dataset/cifar10/'
    test_path = os.path.join(dataset_path, 'test')
    new_test_path = os.path.join(dataset_path, 'testn')
    if not os.path.exists(new_test_path):
        os.mkdir(new_test_path)
    random.seed(12321)
    for label in label_name:
        save_path = os.path.join(new_test_path, label)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        image_path = os.path.join(test_path, label)
        _, _, images = next(os.walk(image_path))
        samples = random.sample(images, 100)
        for image in samples:
            copyfile(os.path.join(image_path, image), os.path.join(save_path, image))


if __name__ == '__main__':
    main()
