import csv
import glob
import os.path

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet50
from torchvision.transforms import transforms

classes = ['Men', 'Women']


class GenderNet(nn.Module):
    def __init__(self):
        super(GenderNet, self).__init__()
        # Load pre-trained ResNet-50
        self.base_model = resnet50(pretrained=True)

        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        # Forward pass through ResNet-50
        x = self.base_model(x)
        return x


if __name__ == '__main__':
    model_path = "../../models/model_gender.pth"
    net = GenderNet()
    net.load_state_dict(torch.load(os.path.abspath(model_path)))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(112),
            transforms.Normalize(mean=(0, 0, 0),
                                 std=(1, 1, 1))
        ]
    )

    image_path = "woman_1010.jpg"
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # Convert BGR image to RGB image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = transform(img).float()
    tensor = tensor.unsqueeze_(0)
    tensor = tensor.to(device)

    with torch.no_grad():
        net.eval()
        outputs = net(tensor)
        # predictions = [1 if result[i] >= 0.5 else 0 for i in range(len(result))]
        # result = classes[predictions[0]]
        _, prediction = torch.max(outputs, 1)
        result = classes[prediction[0]]
        print(result)

        # writer.writerow({'id': os.path.basename(img_path).split('.')[0], 'label': predictions[0]})

        cv2.namedWindow('FRS', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('FRS', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.putText(img, f'Prediction: {result}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow('FRS', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
