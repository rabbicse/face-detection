import argparse
import cv2
import numpy as np
import torch
from screeninfo import get_monitors

from detector.face_detector import FaceDetector


def parse_args():
    parser = argparse.ArgumentParser(description='Image Detection')
    parser.add_argument(
        '--checkpoint', type=str, default='../../models/SCRFD_500M_KPS.pth', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
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
    rz, detections = face_detector.detect(img)

    # draw over image
    draw_img(rz, detections)


def draw_image(frame, boxes, landmarks):
    rect_color = (224, 128, 20)
    circle_color = (255, 255, 128)
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
            cv2.circle(frame, (int(x), int(y)), 3, circle_color, -1, lineType=cv2.LINE_AA)  # Blue dots

    cv2.imshow('Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_img(frame, detections):
    # Get the primary screen dimensions
    monitor = get_monitors()[0]  # Assuming you want the primary monitor
    screen_width = monitor.width
    screen_height = monitor.height
    # Set the window to fullscreen
    # Resize the image to fit the screen while maintaining aspect ratio
    img_height, img_width = frame.shape[:2]
    scale_width = float(screen_width) / float(img_width)
    scale_height = float(screen_height) / float(img_height)
    scale = min(scale_width, scale_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    resized_image = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a black background for padding
    canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # Center the image on the canvas
    y_offset = (screen_height - new_height) // 2
    x_offset = (screen_width - new_width) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    height, width, channels = canvas.shape
    rect_color = (224, 128, 20)
    circle_color = (255, 255, 128)
    # draw bboxes
    for detection in detections:
        bbox = detection['bbox']
        x_min = bbox['x_min'] * width
        y_min = bbox['y_min'] * height
        x_max = bbox['x_max'] * width
        y_max = bbox['y_max'] * height
        cv2.rectangle(canvas, (int(x_min), int(y_min)), (int(x_max), int(y_max)), rect_color, 1)

        cv2.line(canvas, (int(x_min), int(y_min)), (int(x_min + 15), int(y_min)), rect_color, 3)
        cv2.line(canvas, (int(x_min), int(y_min)), (int(x_min), int(y_min + 15)), rect_color, 3)

        cv2.line(canvas, (int(x_max), int(y_max)), (int(x_max - 15), int(y_max)), rect_color, 3)
        cv2.line(canvas, (int(x_max), int(y_max)), (int(x_max), int(y_max - 15)), rect_color, 3)

        cv2.line(canvas, (int(x_max - 15), int(y_min)), (int(x_max), int(y_min)), rect_color, 3)
        cv2.line(canvas, (int(x_max), int(y_min)), (int(x_max), int(y_min + 15)), rect_color, 3)

        cv2.line(canvas, (int(x_min), int(y_max - 15)), (int(x_min), int(y_max)), rect_color, 3)
        cv2.line(canvas, (int(x_min), int(y_max)), (int(x_min + 15), int(y_max)), rect_color, 3)

        # Draw landmarks
        landmarks = detection['landmarks']
        for landmark in landmarks:
            # for (x, y) in landmark:
            x = landmark['x'] * width
            y = landmark['y'] * height
            cv2.circle(canvas, (int(x), int(y)), 3, circle_color, -1, lineType=cv2.LINE_AA)  # Blue dots

    cv2.namedWindow("window", cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('window', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
