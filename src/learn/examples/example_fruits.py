import glob
import os.path
from math import ceil

import cv2
import torch
from torchvision.transforms import transforms

from networks.train_fruits import FruitsNet

classes = ('apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
           'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno',
           'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
           'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato',
           'tomato', 'turnip', 'watermelon')


def draw_result(frame, text):
    height, width, _ = frame.shape

    # get the width and height of the text box
    text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)[0]
    text_x1 = max(int(5 + 10), 0)
    text_y1 = max(int(5 + 10), 0)
    txt_x1 = max(int(5) - int(ceil(2)), 0)
    txt_y1 = max(text_y1 - text_height - int(ceil(2 * 2)) - int(5 * 2), 0)
    txt_x2 = min(txt_x1 + text_width + int(5 * 2), width)
    txt_y2 = min(txt_y1 + text_height + int(5 * 2), height)

    # draw bbox outline
    cv2.rectangle(frame, (int(5), int(5)), (int(5), int(5)), (0, 0, 0), 2)

    # draw text background
    cv2.rectangle(frame, (txt_x1, txt_y1), (txt_x2, txt_y2), (0, 0, 0), cv2.FILLED)

    # draw text
    cv2.putText(frame, text, (text_x1, txt_y1 + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                1, cv2.LINE_AA)


if __name__ == '__main__':
    net = FruitsNet()
    net.load_state_dict(torch.load(os.path.abspath('../models/fruits_resnet50_best.pth')))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224),
            transforms.Normalize(mean=(0, 0, 0),
                                 std=(1, 1, 1))
        ]
    )

    root_dir = os.path.abspath(
        '/mnt/6D4F8771482E7048/Projects/python_projects/face-detection/src/data/fruits/train/')
    for img_path in glob.glob(os.path.join(root_dir, "**/*.*"), recursive=True):
        # print(img_path)
        # img_path = os.path.abspath('../data/input/kitty-cat-kitten-pet-45201.jpeg')
        # load image using PIL
        # img = Image.open(img_path)
        img_orig = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # Convert BGR image to RGB image
        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        tensor = transform(img).float()
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(device)

        with torch.no_grad():
            net.eval()
            outputs = net(tensor)
            # predictions = [1 if result[i] >= 0.5 else 0 for i in range(len(result))]
            # identify max prediction
            _, prediction = torch.max(outputs, 1)
            result = classes[prediction[0]]

            # get vcap property
            MAX_WIDTH = 640
            MAX_HEIGHT = 480
            height, width, channels = img.shape
            ratio = width / height
            w, h = width, height
            if ratio > 1 and w > MAX_WIDTH:  # if width > height then resize based on width
                w = MAX_WIDTH
                h = w / ratio
            elif ratio < 1 and h > MAX_HEIGHT:
                h = MAX_HEIGHT
                w = h * ratio
            # print('{} x {}'.format(w, h))
            img_orig = cv2.resize(img_orig, (int(w), int(h)), interpolation=cv2.INTER_AREA)

            cv2.namedWindow('FRS', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('FRS', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # cv2.putText(img, f'Prediction: {result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
            #             cv2.LINE_AA)
            draw_result(img_orig, result)
            cv2.imshow('FRS', img_orig)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
