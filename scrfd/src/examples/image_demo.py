import argparse
import cv2
import torch

from detector.face_detector import FaceDetector


def parse_args():
    parser = argparse.ArgumentParser(description='Image Detection')
    parser.add_argument(
        '--checkpoint', type=str, default='../../models/SCRFD_500M_KPS.pth', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cpu', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
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

    # forward
    resized_img, result = face_detector.detect(img)

    draw_img(resized_img, result['bbox'], result['keypoints'])


def draw_img(frame, boxes, landmarks):
    rect_color = (224, 128, 20)
    circle_color = (64, 128, 64)
    # draw bboxes
    for box in boxes:
        x_min, y_min, x_max, y_max, _ = box
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), rect_color, 1)

        cv2.line(frame, (int(x_min), int(y_min)), (int(x_min + 15), int(y_min)), rect_color, 3)
        cv2.line(frame, (int(x_min), int(y_min)), (int(x_min), int(y_min + 15)), rect_color, 3)

        cv2.line(frame, (int(x_max), int(y_max)), (int(x_max - 15), int(y_max)), rect_color, 3)
        cv2.line(frame, (int(x_max), int(y_max)), (int(x_max), int(y_max - 15)), rect_color, 3)

        cv2.line(frame, (int(x_max - 15), int(y_min)), (int(x_max), int(y_min)), rect_color, 3)
        cv2.line(frame, (int(x_max), int(y_min)), (int(x_max), int(y_min + 15)), rect_color, 3)

        cv2.line(frame, (int(x_min), int(y_max - 15)), (int(x_min), int(y_max)), rect_color, 3)
        cv2.line(frame, (int(x_min), int(y_max)), (int(x_min + 15), int(y_max)), rect_color, 3)

    # Draw landmarks
    for landmark in landmarks:
        for (x, y) in landmark:
            cv2.circle(frame, (int(x), int(y)), 3, circle_color, -1)  # Blue dots

    cv2.imshow('Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
