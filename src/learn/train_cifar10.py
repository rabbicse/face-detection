import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn, optim
from torchvision import transforms

from resnet import ResNet50


class ResnetTrain:
    def __init__(self):
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.train_loader = None
        self.test_loader = None

        # set hyper parameters
        self.batch_size = 128
        # Learning rate
        self.lr = 0.001
        # Number of training epochs
        self.num_epochs = 100

        # self.net = ResNet([2, 2, 2, 2]) # resnet 18
        self.net = ResNet50()  # resnet 50
        # fine tune
        # Freeze the layers
        for param in self.net.parameters():
            param.requires_grad = False
        # Change the last layer to cifar10 number of output classes.
        # Also unfreeze the penultimate layer. We will finetune just these two layers.
        self.net.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, weight_decay=0.001, momentum=0.9)

    def load_dataset(self):
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
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
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            running_corrects = 0
            print(f'Total train set size: {len(self.train_loader.dataset)}')
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
                running_loss += loss.item()

            epoch_loss = running_loss / len(self.train_loader.dataset)

            print('Epoch: {} Loss: {:.4f}'.format(epoch, epoch_loss))

        print('Finished Training')

        PATH = './cifar10_net.pth'
        torch.save(self.net.state_dict(), PATH)


if __name__ == '__main__':
    # init train
    trainer = ResnetTrain()
    trainer.load_dataset()
    trainer.train_data()
