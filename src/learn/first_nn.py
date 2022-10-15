import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt


class FirstNN(nn.Module):
    def __init__(self):
        super(FirstNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, (3, 3))
        self.conv2 = nn.Conv2d(10, 5, (3, 3))

    def forward(self, x):
        model = nn.Sequential(self.conv1, self.conv2)
        return model(x)


class FirstTrain:
    def __init__(self):
        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', ' truck')
        self.train_loader = None
        self.data_iter = None

    def load_dataset(self):
        train_set = torchvision.datasets.ImageFolder(root='./data/cifar10/train',
                                                     transform=torchvision.transforms.ToTensor())
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=4)

        self.data_iter = iter(self.train_loader)

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


if __name__ == '__main__':
    # init cnn
    cnn = FirstNN()

    # init train
    trainer = FirstTrain()
    trainer.load_dataset()
    images, labels = trainer.iter_data()

    out = cnn(images)

    trainer.plot_image(trainer.convert_tensor_to_numpy(out))
