import cv2 as cv2
import torch
from torchvision.transforms import transforms

from resnet_18 import Resnet

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net = Resnet(3, 10)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
net.load_state_dict(torch.load('./cifar_net.pth'))

data = cv2.imread('./data/cifar10/test/airplane/0001.png', cv2.IMREAD_UNCHANGED)
# Convert BGR image to RGB image
data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGB)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

with torch.no_grad():
    tensor = transform(data)
    tensor = torch.reshape(tensor, (1, 3, 112, 112))

    result = net(tensor.cuda())
    print(result)
    _, predicted = torch.max(result, 1)
    print(predicted)
    print(classes[predicted[0]])
