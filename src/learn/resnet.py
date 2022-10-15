import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from torch import optim
from torchvision import transforms


class ResidualBlock(nn.Module):
    """
    Source: https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
    """

    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.down_sample = down_sample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.down_sample:
            residual = self.down_sample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResnetTrain:
    def __init__(self):
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.train_loader = None
        self.test_loader = None
        self.batch_size = 1
        # self.net = ResNet(ResidualBlock, [3, 4, 6, 3])

        self.net = ResNet(ResidualBlock, [2, 2, 2, 2])

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=1e-4)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, weight_decay=0.001, momentum=0.9)

    def load_dataset(self):
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )

        # define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
        # transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #      ])

        train_set = torchvision.datasets.ImageFolder(root='./data/cifar10/train', transform=transform)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size,
                                                        shuffle=True, num_workers=8)

        test_set = torchvision.datasets.ImageFolder(root='./data/cifar10/test', transform=transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                                       shuffle=False, num_workers=8)

    def iter_data(self):
        return self.data_iter.next()

    def convert_tensor_to_numpy(self, tensor):
        print(tensor.shape)

        # image after convolution
        sample = tensor[0, 0, :, :]
        print(sample.shape)
        return sample.detach().numpy()

    def plot_image(self, image):
        plt.figure(figsize=(2, 2))
        plt.imshow(image)
        plt.show()

    def train_data(self):
        torch.cuda.empty_cache()
        for epoch in range(1):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0], data[1]

                inputs = inputs.to(self.device)

                labels = labels.to(self.device)

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

        PATH = './cifar_net.pth'
        torch.save(self.net.state_dict(), PATH)


if __name__ == '__main__':
    # init train
    trainer = ResnetTrain()
    trainer.load_dataset()
    trainer.train_data()