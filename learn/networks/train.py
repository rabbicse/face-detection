import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from network import Network


class Train:
    def __init__(self):
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.train_loader = None
        self.test_loader = None
        self.batch_size = 4
        self.net = Network()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    def load_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_set = torchvision.datasets.ImageFolder(root='./data/cifar10/train', transform=transform)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size,
                                                        shuffle=True, num_workers=8)

        test_set = torchvision.datasets.ImageFolder(root='./data/cifar10/test', transform=transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                                       shuffle=False, num_workers=8)

    def imshow(self, img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def show_images(self):
        # get some random training images
        dataiter = iter(self.train_loader)
        images, labels = dataiter.next()
        print(type(images))

        # show images
        self.imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join(f'{self.classes[labels[j]]:5s}' for j in range(self.batch_size)))

    def train_data(self):
        torch.cuda.empty_cache()
        for epoch in range(10):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')

        PATH = '../models/cifar_net.pth'
        torch.save(self.net.state_dict(), PATH)

    def test_data(self):
        dataiter = iter(self.test_loader)
        images, labels = dataiter.next()
        print(images.shape)
        print(type(images))

        # print images
        self.imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join(f'{self.classes[labels[j]]:5s}' for j in range(1)))

        self.net.load_state_dict(torch.load('./cifar_net.pth'))
        outputs = self.net(images.cuda())
        print(outputs)
        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join(f'{self.classes[predicted[j]]:5s}'
                                      for j in range(1)))

        # correct = 0
        # total = 0
        # # since we're not training, we don't need to calculate the gradients for our outputs
        # with torch.no_grad():
        #     for data in self.test_loader:
        #         images, labels = data
        #         # calculate outputs by running images through the network
        #         outputs = self.net(images.cuda())
        #         # the class with the highest energy is what we choose as prediction
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels.cuda()).sum().item()
        #
        # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
