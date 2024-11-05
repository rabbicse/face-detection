import glob
import os.path
import warnings

import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms
warnings.filterwarnings("ignore")


def get_model_instance_segmentation(num_classes, pretrained=True):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


if __name__ == '__main__':
    net = get_model_instance_segmentation(2)
    net.load_state_dict(torch.load(os.path.abspath(
        '/mnt/6D4F8771482E7048/Projects/python_projects/face-detection/src/learn/models/rickshaw_net.pth')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ]
    )

    root_dir = os.path.abspath('/mnt/6D4F8771482E7048/Projects/python_projects/face-detection/src/data/rickshaw_data/')
    # img_path = os.path.join(root_dir, '20210814_173844.jpg')#'ups-1586092274642.jpg')#'20210814_173844.jpg')

    for img_path in glob.glob(os.path.join(root_dir, "*.jpg")):
        # print(img_path)
        # img_path = os.path.abspath('../data/input/kitty-cat-kitten-pet-45201.jpeg')
        # load image using PIL
        # img = Image.open(img_path)
        img = cv2.imread(img_path)

        # cv2.imshow('Rickshaw', img)
        # cv2.waitKey(0)

        # Convert BGR image to RGB image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Rickshaw', img)
        # cv2.waitKey(0)

        tensor = transform(img).float()
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(device)

        with torch.no_grad():
            net.eval()
            result = net(tensor)
            for r in result:
                print(r)

                for i in range(len(r['boxes'].cpu().tolist())):
                    box = r['boxes'][i]
                    scores = r['scores']
                    if scores[i] < 0.6:
                        continue

                    img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)
            # predictions = [1 if result[i] >= 0.5 else 0 for i in range(len(result))]
            result = 'Rickshaw'#classes[predictions[0]]

            # writer.writerow({'id': os.path.basename(img_path).split('.')[0], 'label': predictions[0]})

            cv2.namedWindow('FRS', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('FRS', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.putText(img, f'Prediction: {result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow('FRS', img)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

        # break
