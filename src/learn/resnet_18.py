import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from torch import optim
from torchvision import transforms


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, identity_down_sample=None):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=1)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_down_sample = identity_down_sample

        self.conv01 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv02 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        #
        # x = self.conv2(x)
        # x = self.bn2(x)
        # if self.identity_down_sample is not None:
        #     identity = self.identity_down_sample(identity)
        # x += identity
        # x = self.relu(x)
        # return x

        out = self.conv01(x)
        out = self.conv02(out)
        if self.identity_down_sample is not None:
            identity = self.identity_down_sample(identity)
        out += identity
        out = self.relu(out)
        return out


class Resnet(nn.Module):
    def __init__(self, in_channels, num_classes, output_channels=64):
        super(Resnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=output_channels, kernel_size=7, padding=3,
                               stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self.__make_resnet_layer(in_channels=64, out_channels=64, stride=1)
        self.layer2 = self.__make_resnet_layer(in_channels=64, out_channels=128, stride=2)
        self.layer3 = self.__make_resnet_layer(in_channels=128, out_channels=256, stride=2)
        self.layer4 = self.__make_resnet_layer(in_channels=256, out_channels=512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def identity_down_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def __make_resnet_layer(self, in_channels, out_channels, stride):
        identity_down_sample = None
        if stride != 1:
            identity_down_sample = self.identity_down_sample(in_channels, out_channels)
        return nn.Sequential(
            ResnetBlock(in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        identity_down_sample=identity_down_sample),
            ResnetBlock(in_channels=out_channels,
                        out_channels=out_channels)
        )


class ResnetTrain:
    def __init__(self):
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.train_loader = None
        self.test_loader = None
        self.batch_size = 256
        self.net = Resnet(3, 10)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=1e-4)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    def load_dataset(self):
        transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

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
        for epoch in range(100):  # loop over the dataset multiple times

            running_loss = 0.0
            running_corrects = 0
            # print(f'Total train set size: {len(self.train_loader.dataset)}')
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # print(f'Input size: {len(inputs)}')

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # print(f'Batch index: {i}')

            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = running_corrects.double() / len(self.train_loader.dataset)

            print('Epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

            if epoch_loss < 0.05:
                print(f'Found expected accuracy!')
                break

        print('Finished Training')

        PATH = './cifar_net.pth'
        torch.save(self.net.state_dict(), PATH)


def train_data():
    model = Resnet(in_channels=1, num_classes=10)
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


if __name__ == '__main__':
    # init train
    trainer = ResnetTrain()
    trainer.load_dataset()
    trainer.train_data()
