import copy
import os
import random
import shutil

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils import data
from torchvision import transforms, datasets

from resnet import ResNet50


class CubNet(ResNet50):
    def __init__(self):
        super(CubNet, self).__init__()
        fc = nn.Linear(self.fc.in_features, 200)
        self.fc = fc


class CubTrain:
    def __init__(self):
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None

        # set hyper parameters
        self.img_size = 224
        self.means = [0.485, 0.456, 0.406]  # [0.4857, 0.4991, 0.4312]
        self.stds = [0.229, 0.224, 0.225] # [0.1824, 0.1813, 0.1932]

        self.batch_size = 16

        # Number of training epochs
        self.num_epochs = 20

        # Learning rate
        self.lr = 1e-7

        # construct resnet 50
        self.net = CubNet()  # resnet 50

        # set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        # set optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        # set criterion to calculate loss
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)

        seed = 1234

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def load_dataset(self):
        train_transforms = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(self.img_size, padding=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.means,
                                 std=self.stds)
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.means,
                                 std=self.stds)
        ])

        train_data = torchvision.datasets.ImageFolder(root='../data/CUB_200_2011/CUB_200_2011/train',
                                                      transform=train_transforms)
        test_data = torchvision.datasets.ImageFolder(root='../data/CUB_200_2011/CUB_200_2011/test',
                                                     transform=test_transforms)

        valid_ratio = 0.9
        n_train_examples = int(len(train_data) * valid_ratio)
        n_valid_examples = len(train_data) - n_train_examples

        train_data, valid_data = data.random_split(train_data,
                                                   [n_train_examples, n_valid_examples])
        valid_data = copy.deepcopy(valid_data)
        valid_data.dataset.transform = test_transforms

        print(f'Number of training examples: {len(train_data)}')
        print(f'Number of validation examples: {len(valid_data)}')
        print(f'Number of testing examples: {len(test_data)}')

        self.train_iterator = data.DataLoader(train_data,
                                              shuffle=True,
                                              num_workers=8,
                                              batch_size=self.batch_size)

        self.valid_iterator = data.DataLoader(valid_data,
                                              num_workers=8,
                                              batch_size=self.batch_size)

        self.test_iterator = data.DataLoader(test_data,
                                             num_workers=8,
                                             batch_size=self.batch_size)

    def normalize_image(self, image):
        image_min = image.min()
        image_max = image.max()
        image.clamp_(min=image_min, max=image_max)
        image.add_(-image_min).div_(image_max - image_min + 1e-5)
        return image

    def plot_images(self, images, labels, classes, normalize=True):

        n_images = len(images)

        rows = int(np.sqrt(n_images))
        cols = int(np.sqrt(n_images))

        fig = plt.figure(figsize=(15, 15))

        for i in range(rows * cols):

            ax = fig.add_subplot(rows, cols, i + 1)

            image = images[i]

            if normalize:
                image = self.normalize_image(image)

            ax.imshow(image.permute(1, 2, 0).cpu().numpy())
            label = classes[labels[i]]
            ax.set_title(label)
            ax.axis('off')

    def format_label(self, label):
        label = label.split('.')[-1]
        label = label.replace('_', ' ')
        label = label.title()
        label = label.replace(' ', '')
        return label

    def calculate_topk_accuracy(self, y_pred, y, k=5):
        with torch.no_grad():
            batch_size = y.shape[0]
            _, top_pred = y_pred.topk(k, 1)
            top_pred = top_pred.t()
            correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
            correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            acc_1 = correct_1 / batch_size
            acc_k = correct_k / batch_size
        return acc_1, acc_k

    def train(self, iterator):
        epoch_loss = 0
        epoch_acc_1 = 0
        epoch_acc_5 = 0

        for (x, y) in iterator:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.net(x)

            loss = self.criterion(outputs, y)

            acc_1, acc_5 = self.calculate_topk_accuracy(outputs, y)

            loss.backward()

            self.optimizer.step()

            # scheduler.step()

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()

        epoch_loss /= len(iterator)
        epoch_acc_1 /= len(iterator)
        epoch_acc_5 /= len(iterator)

        return epoch_loss, epoch_acc_1, epoch_acc_5

    def evaluate(self, iterator):

        epoch_loss = 0
        epoch_acc_1 = 0
        epoch_acc_5 = 0

        self.net.eval()

        with torch.no_grad():
            for (x, y) in iterator:
                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.net(x)

                loss = self.criterion(outputs, y)

                acc_1, acc_5 = self.calculate_topk_accuracy(outputs, y)

                epoch_loss += loss.item()
                epoch_acc_1 += acc_1.item()
                epoch_acc_5 += acc_5.item()

        epoch_loss /= len(iterator)
        epoch_acc_1 /= len(iterator)
        epoch_acc_5 /= len(iterator)

        return epoch_loss, epoch_acc_1, epoch_acc_5

    def train_data(self):
        best_valid_loss = float('inf')
        torch.cuda.empty_cache()
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            train_loss, train_acc_1, train_acc_5 = self.train(self.train_iterator)
            valid_loss, valid_acc_1, valid_acc_5 = self.evaluate(self.valid_iterator)

            print(
                f'Train Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1 * 100:6.2f}% | Train Acc @5: {train_acc_5 * 100:6.2f}%')
            print(
                f'Valid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1 * 100:6.2f}% | Valid Acc @5: {valid_acc_5 * 100:6.2f}%')

        print('Finished Training')

        # save model
        model_path = '../models/cub_200_2011_net.pth'
        torch.save(self.net.state_dict(), model_path)


def prepare_train_data():
    ROOT = 'data/'
    TRAIN_RATIO = 0.8

    data_dir = os.path.join(ROOT, 'CUB_200_2011/CUB_200_2011')
    images_dir = os.path.join(data_dir, 'images')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    os.makedirs(train_dir)
    os.makedirs(test_dir)

    classes = os.listdir(images_dir)

    for c in classes:

        class_dir = os.path.join(images_dir, c)

        images = os.listdir(class_dir)

        n_train = int(len(images) * TRAIN_RATIO)

        train_images = images[:n_train]
        test_images = images[n_train:]

        os.makedirs(os.path.join(train_dir, c), exist_ok=True)
        os.makedirs(os.path.join(test_dir, c), exist_ok=True)

        for image in train_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(train_dir, c, image)
            shutil.copyfile(image_src, image_dst)

        for image in test_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(test_dir, c, image)
            shutil.copyfile(image_src, image_dst)

    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=transforms.ToTensor())

    means = torch.zeros(3)
    stds = torch.zeros(3)

    for img, label in train_data:
        means += torch.mean(img, dim=(1, 2))
        stds += torch.std(img, dim=(1, 2))

    means /= len(train_data)
    stds /= len(train_data)

    print(f'Calculated means: {means}')
    print(f'Calculated stds: {stds}')


if __name__ == '__main__':
    # prepare_train_data()
    # init train
    trainer = CubTrain()
    trainer.load_dataset()
    trainer.train_data()
