import gc
import os
import time
import warnings

import torch
import torchvision
from PIL import Image
from torch import nn, optim
from torch.utils import data
from torchvision import transforms

from .resnet import ResNet50

warnings.filterwarnings("ignore")


def custom_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')


class FruitsNet(ResNet50):
    def __init__(self):
        super(FruitsNet, self).__init__()
        self.fc = nn.Linear(self.fc.in_features, 36)


class FruitsTrain:
    def __init__(self):
        self.classes = ('apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
                        'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno',
                        'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
                        'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato',
                        'tomato', 'turnip', 'watermelon')
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None

        # set hyper parameters
        self.img_size = 224
        self.means = (0, 0, 0)
        self.stds = (1, 1, 1)

        self.batch_size = 12

        # Number of training epochs
        self.num_epochs = 200

        # Learning rate
        self.lr = 0.0001

        # Initiate net
        self.net = FruitsNet()

        # set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        # set optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        # set criterion to calculate loss
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)

        # Learning Rate Scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def load_dataset(self):
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.means,
                                 std=self.stds)
        ])

        test_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.means,
                                 std=self.stds)
        ])

        train_dataset = torchvision.datasets.ImageFolder(root=os.path.abspath('../data/fruits/train'),
                                                         transform=train_transforms)
        validation_dataset = torchvision.datasets.ImageFolder(root=os.path.abspath('../data/fruits/validation'),
                                                              transform=test_transforms)
        test_dataset = torchvision.datasets.ImageFolder(root=os.path.abspath('../data/fruits/test'),
                                                        transform=test_transforms)

        print(train_dataset.classes)

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

        print('Load data done!')

    def train(self, iterator):
        gc.collect()
        torch.cuda.empty_cache()

        # Local Parameters
        epoch_loss = 0
        epoch_acc = 0
        start_time = time.time()

        # Iterating over data loader
        for images, labels in iterator:
            # Loading images and labels to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Reseting Gradients
            self.optimizer.zero_grad()

            # Forward
            outputs = self.net(images)

            # identify max prediction
            _, prediction = torch.max(outputs, 1)

            # Calculating Loss
            loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            # append losses
            epoch_loss += loss.item() * images.size(0)

            # append accuracy
            epoch_acc += torch.sum(prediction == labels.data)

            del images
            del labels

        # Overall Epoch Results
        end_time = time.time()
        total_time = end_time - start_time

        # Acc and Loss
        # epoch_loss = np.mean(epoch_loss)
        # epoch_acc = np.mean(epoch_acc)
        # self.lr_scheduler.step()
        return epoch_loss / len(iterator.dataset), epoch_acc / len(iterator.dataset), total_time

    def evaluate(self, iterator, best_val_acc, mode='test'):
        gc.collect()
        torch.cuda.empty_cache()

        # Local Parameters
        epoch_loss = 0
        epoch_acc = 0
        start_time = time.time()

        # Iterating over data loader
        for images, labels in iterator:
            # Loading images and labels to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward
            outputs = self.net(images)

            # identify max prediction
            _, prediction = torch.max(outputs, 1)

            # Calculating Loss
            loss = self.criterion(outputs, labels)

            # Calculate loss
            epoch_loss += loss.item() * images.size(0)

            # Calculating Accuracy
            epoch_acc += torch.sum(prediction == labels.data)

            del images
            del labels

        # Overall Epoch Results
        end_time = time.time()
        total_time = end_time - start_time

        # Saving best model
        if epoch_acc > best_val_acc and mode == 'val':
            best_val_acc = epoch_acc
            torch.save(self.net.state_dict(), os.path.abspath('models/fruits_resnet50_best.pth'))

        return epoch_loss / len(iterator.dataset), epoch_acc / len(iterator.dataset), total_time, best_val_acc

    def train_data(self):
        best_val_acc = 0
        torch.cuda.empty_cache()

        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            # Training
            print("Training")
            loss, acc, elapsed = self.train(self.train_iterator)
            # Print Epoch Details
            print(f'Epoch {epoch + 1} Loss : {loss} Acc : {acc * 100}% Time : {elapsed}')

            # Validation
            print("Validating")
            loss, acc, elapsed, best_val_acc = self.evaluate(self.valid_iterator, best_val_acc=best_val_acc, mode='val')
            # Print Epoch Details
            print(f'Epoch {epoch + 1} Loss : {loss} Acc : {acc * 100}:.3f% Time : {elapsed}')

            # Test
            # print("\nTesting")
            # loss, acc, elapsed, best_val_acc = self.evaluate(self.test_iterator, best_val_acc=best_val_acc)
            # # Print Epoch Details
            # print(f'Epoch {epoch + 1} Loss : {loss} Acc : {acc} Time : {elapsed}')

        print('Finished Training')

        # save model
        model_path = os.path.abspath('models/fruits_net.pth')
        torch.save(self.net.state_dict(), model_path)


if __name__ == '__main__':
    # prepare_train_data()
    # init train
    trainer = FruitsTrain()
    trainer.load_dataset()
    trainer.train_data()
