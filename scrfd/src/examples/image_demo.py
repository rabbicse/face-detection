import argparse
import cv2
import numpy as np
import torch
from screeninfo import get_monitors

from detector.face_detector import FaceDetector
from detector.utils.cv_utils import draw_img, resize_image_to_monitor


def parse_args():
    parser = argparse.ArgumentParser(description='Image Detection')
    parser.add_argument(
        '--checkpoint', type=str, default='../../models/SCRFD_500M_KPS.pth', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cpu', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # device
    device = torch.device(args.device)

    # initialize detector
    face_detector = FaceDetector(args.checkpoint, device=device)

    # read image
    img = cv2.imread('messi-hair.jpg')
    # img = cv2.imread('selfie.jpg')

    # forward
    detections = face_detector.detect(img)

    # draw over image
    canvas = resize_image_to_monitor(img)
    draw_img(canvas, detections, is_fullscreen=True)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
