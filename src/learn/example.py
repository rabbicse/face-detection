import cv2 as cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn.functional import normalize
from torchvision.transforms import transforms

from network import Network

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net = Network()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
net.load_state_dict(torch.load('./cifar_net.pth'))

data = cv2.imread('./data/cifar10/test/airplane/0001.png', cv2.IMREAD_UNCHANGED)
# Convert BGR image to RGB image
data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGB)

# print(type(data))

# data = np.transpose(data, (1, 2, 0))

# print(data.shape)
# cv2.imshow('input', data)
# cv2.waitKey(0)

# data = data.reshape(1, 3, 32, 32)
# print(type(data))
# print(data.shape)


# data = Image.open('./data/cifar10/test/cat/0001.png')

# data = np.asarray(data)
# transform = transforms.Compose(
#     [
#         transforms.PILToTensor(),
#         transforms.ConvertImageDtype(torch.float32),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

with torch.no_grad():
    # tensor = torch.from_numpy(np.float32(data))
    # tensor = transform(np.float32(data))

    # data = np.float32(data)
    tensor = transform(data)
    tensor = torch.reshape(tensor, (1, 3, 32, 32))
    # tensor = transform(tensor.cuda())
    # print(tensor.shape)

    # print(type(tensor))
    # print(tensor)

    result = net(tensor.cuda())
    print(result)
    _, predicted = torch.max(result, 1)
    print(predicted)
    print(classes[predicted[0]])
