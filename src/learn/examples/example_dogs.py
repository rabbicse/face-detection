import csv
import glob
import os.path

import cv2
import torch
from PIL import Image
from torchvision.transforms import transforms
from networks.train_dogs import DogCatNet

classes = ['dog', 'cat']

if __name__ == '__main__':
    net = DogCatNet()
    net.load_state_dict(torch.load(os.path.abspath('models/dog_cat_resnet50_best.pth')))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize(mean=(0, 0, 0),
                                 std=(1, 1, 1))
        ]
    )

    root_dir = os.path.abspath('../data/dogs-vs-cats/test')

    # with open('submission.csv', 'w+', newline='', encoding='utf-8') as f:
    #     writer = csv.DictWriter(f, fieldnames=['id', 'label'], quoting=csv.QUOTE_ALL)
    #     writer.writerow({'id': 'Id', 'label': 'label'})
    for img_path in glob.glob(os.path.join(root_dir, "*.*")):
        # print(img_path)
        # img_path = os.path.abspath('../data/input/kitty-cat-kitten-pet-45201.jpeg')
        # load image using PIL
        # img = Image.open(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # Convert BGR image to RGB image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tensor = transform(img).float()
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(device)

        with torch.no_grad():
            net.eval()
            result = net(tensor)
            predictions = [1 if result[i] >= 0.5 else 0 for i in range(len(result))]
            result = classes[predictions[0]]

            # writer.writerow({'id': os.path.basename(img_path).split('.')[0], 'label': predictions[0]})

            cv2.namedWindow('FRS', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('FRS', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.putText(img, f'Prediction: {result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow('FRS', img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

        # break
