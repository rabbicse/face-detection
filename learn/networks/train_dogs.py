import gc
import os
import time

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .resnet import ResNet50


class DogCatNet(ResNet50):
    def __init__(self):
        super(DogCatNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048, 1, bias=True),
            nn.Sigmoid()
        )


class DogCatDataset(Dataset):

    def __init__(self, imgs, class_to_int, mode="train", transforms=None, root_path='../data/dogs-vs-cats/train'):

        super().__init__()
        self.imgs = imgs
        self.class_to_int = class_to_int
        self.mode = mode
        self.transforms = transforms
        self.dir_train = os.path.abspath(root_path)

    def __getitem__(self, idx):

        image_name = self.imgs[idx]
        img = Image.open(os.path.join(self.dir_train, image_name))
        img = img.resize((224, 224))

        if self.mode == "train" or self.mode == "val":
            # Preparing class label
            label = self.class_to_int(image_name.split(".")[0])
            label = torch.tensor(label, dtype=torch.float32)

            # Apply Transforms on image
            img = self.transforms(img)

            return img, label

        elif self.mode == "test":
            # Apply Transforms on image
            img = self.transforms(img)

            return img

    def __len__(self):
        return len(self.imgs)


class DogCatTrain:
    def __init__(self):
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None

        # set hyper parameters
        self.img_size = 224
        self.means = (0, 0, 0)
        self.stds = (1, 1, 1)

        self.batch_size = 8

        # Number of training epochs
        self.num_epochs = 10

        # Learning rate
        self.lr = 0.0001

        # Initiate net
        self.net = DogCatNet()

        # set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        # set optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        # set criterion to calculate loss
        self.criterion = nn.BCELoss()
        self.criterion.to(self.device)

        # Learning Rate Scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def load_dataset(self):
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomCrop(204),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.means,
                                 std=self.stds)
        ])

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.means,
                                 std=self.stds)
        ])

        train_images = os.listdir(os.path.abspath('../data/dogs-vs-cats/train'))
        test_images = os.listdir(os.path.abspath('../data/dogs-vs-cats/test'))

        train_data, valid_data = train_test_split(train_images, test_size=0.20)

        train_dataset = DogCatDataset(train_data, class_to_int=self.class_to_int, mode='train',
                                      transforms=train_transforms)
        validation_dataset = DogCatDataset(valid_data, class_to_int=self.class_to_int, mode='val',
                                           transforms=test_transforms)
        test_dataset = DogCatDataset(test_images, class_to_int=self.class_to_int, mode='test',
                                     transforms=test_transforms, root_path=os.path.abspath('../data/dogs-vs-cats/test'))

        self.train_iterator = data.DataLoader(dataset=train_dataset,
                                              shuffle=True,
                                              num_workers=8,
                                              batch_size=self.batch_size)

        self.valid_iterator = data.DataLoader(dataset=validation_dataset,
                                              shuffle=True,
                                              num_workers=8,
                                              batch_size=self.batch_size)

        self.test_iterator = data.DataLoader(dataset=test_dataset,
                                             shuffle=False,
                                             num_workers=8,
                                             batch_size=self.batch_size)

    @staticmethod
    def class_to_int(x: str):
        return 0 if 'dog' in x.lower() else 1

    @staticmethod
    def accuracy(predictions, trues):
        """
        :param predictions:
        :param trues:
        :return:
        """
        # Converting predictions to 0 or 1
        predictions = [1 if predictions[i] >= 0.5 else 0 for i in range(len(predictions))]

        # Calculating accuracy by comparing predictions with true labels
        acc = [1 if predictions[i] == trues[i] else 0 for i in range(len(predictions))]

        # Summing over all correct predictions
        acc = np.sum(acc) / len(predictions)

        return acc * 100

    def plot_images(self, images, labels, classes, normalize=True):
        """
        :param images:
        :param labels:
        :param classes:
        :param normalize:
        :return:
        """

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

    @staticmethod
    def format_label(label):
        label = label.split('.')[-1]
        label = label.replace('_', ' ')
        label = label.title()
        return label.replace(' ', '')

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

    def train(self, iterator: DataLoader):
        gc.collect()
        torch.cuda.empty_cache()

        # Local Parameters
        epoch_loss = []
        epoch_acc = []
        start_time = time.time()

        # Iterating over data loader
        for images, labels in iterator:
            # Loading images and labels to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            labels = labels.reshape((labels.shape[0], 1))  # [N, 1] - to match with preds shape

            # Reseting Gradients
            self.optimizer.zero_grad()

            # Forward
            preds = self.net(images)

            # Calculating Loss
            _loss = self.criterion(preds, labels)
            loss = _loss.item()
            epoch_loss.append(loss)

            # Calculating Accuracy
            acc = self.accuracy(preds, labels)
            epoch_acc.append(acc)

            # Backward
            _loss.backward()
            self.optimizer.step()

            del images
            del labels

        # Overall Epoch Results
        end_time = time.time()
        total_time = end_time - start_time

        # Acc and Loss
        epoch_loss = np.mean(epoch_loss)
        epoch_acc = np.mean(epoch_acc)

        return epoch_loss, epoch_acc, total_time

    def evaluate(self, iterator, best_val_acc, mode='test'):
        gc.collect()
        torch.cuda.empty_cache()

        # Local Parameters
        epoch_loss = []
        epoch_acc = []
        start_time = time.time()

        # Iterating over data loader
        for images, labels in iterator:
            # Loading images and labels to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            labels = labels.reshape((labels.shape[0], 1))  # [N, 1] - to match with preds shape

            # Forward
            preds = self.net(images)

            # Calculating Loss
            _loss = self.criterion(preds, labels)
            loss = _loss.item()
            epoch_loss.append(loss)

            # Calculating Accuracy
            acc = self.accuracy(preds, labels)
            epoch_acc.append(acc)

            del images
            del labels

        # Overall Epoch Results
        end_time = time.time()
        total_time = end_time - start_time

        # Acc and Loss
        epoch_loss = np.mean(epoch_loss)
        epoch_acc = np.mean(epoch_acc)

        # Saving best model
        if epoch_acc > best_val_acc and mode == 'val':
            best_val_acc = epoch_acc
            torch.save(self.net.state_dict(), os.path.abspath('models/dog_cat_resnet50_best.pth'))

        return epoch_loss, epoch_acc, total_time, best_val_acc

    def train_data(self):
        best_val_acc = 0
        torch.cuda.empty_cache()

        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            # Training
            print(f'Training epoch: {epoch + 1}')
            loss, acc, _time = self.train(self.train_iterator)
            # Print Epoch Details
            print(f'Epoch: {epoch + 1} Loss : {loss} Acc : {acc} Time: {_time}')

            # Validation
            print(f'Validating epoch: {epoch + 1}')
            loss, acc, _time, best_val_acc = self.evaluate(self.valid_iterator, best_val_acc=best_val_acc, mode='val')
            # Print Epoch Details
            print(f'Epoch: {epoch + 1} Loss : {loss} Acc : {acc} Time: {_time}')

            # Test
            # print(f'Testing epoch: {epoch + 1}')
            # loss, acc, _time, best_val_acc = self.evaluate(self.test_iterator, best_val_acc=best_val_acc)
            # # Print Epoch Details
            # print(f'Epoch: {epoch + 1} Loss : {loss} Acc : {acc} Time: {_time}')

        print('Finished Training')

        # save model
        model_path = os.path.abspath('models/dog_cat_net.pth')
        torch.save(self.net.state_dict(), model_path)


if __name__ == '__main__':
    # prepare_train_data()
    # init train
    trainer = DogCatTrain()
    trainer.load_dataset()
    trainer.train_data()
