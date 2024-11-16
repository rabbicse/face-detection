import argparse

import cv2
import numpy as np
import torch

import configs.scrfd.scrfd_500m_bnkps as conf
from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    # parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()
    args.config = '../configs/scrfd/scrfd_500m_bnkps.py'
    args.checkpoint = '../model.pth'  # 'model_1_kps.onnx',
    print(args)

    device = torch.device(args.device)

    model = init_detector(args.config, args.checkpoint, device=device)

    camera = cv2.VideoCapture(args.camera_id)

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, img = camera.read()
        result = inference_detector(model, img)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        draw_img(img, result['bbox'], result['keypoints'])

        # # print(result)
        # bbox_result = result['bboxes']
        # bboxes = np.vstack(bbox_result)
        # labels = [
        #     np.full(bbox.shape[0], i, dtype=np.int32)
        #     for i, bbox in enumerate(bbox_result)
        # ]
        # labels = np.concatenate(labels)
        # scores = bboxes[:, -1]
        # inds = scores > 0.5
        # bboxes = bboxes[inds, :]
        # keypoints = result['keypoints']
        # kp = np.vstack(keypoints)
        # kp = kp[inds, :]

        # image_with_detections = draw_detections(img, [result])
        # Display the image
        # cv2.imshow("Detections", image_with_detections)

        # model.show_result(
        #     img, result['bboxes'], score_thr=args.score_thr, wait_time=1, show=True)


def draw_img(frame, boxes, landmarks):
    # boxes = boxes.astype('int32')

    # draw bboxes
    for box in boxes:
        x_min, y_min, x_max, y_max, _ = box
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 255), 1)

        cv2.line(frame, (int(x_min), int(y_min)), (int(x_min + 15), int(y_min)), (255, 0, 255), 3)
        cv2.line(frame, (int(x_min), int(y_min)), (int(x_min), int(y_min + 15)), (255, 0, 255), 3)

        cv2.line(frame, (int(x_max), int(y_max)), (int(x_max - 15), int(y_max)), (255, 0, 255), 3)
        cv2.line(frame, (int(x_max), int(y_max)), (int(x_max), int(y_max - 15)), (255, 0, 255), 3)

        cv2.line(frame, (int(x_max - 15), int(y_min)), (int(x_max), int(y_min)), (255, 0, 255), 3)
        cv2.line(frame, (int(x_max), int(y_min)), (int(x_max), int(y_min + 15)), (255, 0, 255), 3)

        cv2.line(frame, (int(x_min), int(y_max - 15)), (int(x_min), int(y_max)), (255, 0, 255), 3)
        cv2.line(frame, (int(x_min), int(y_max)), (int(x_min + 15), int(y_max)), (255, 0, 255), 3)

    # Draw landmarks
    for landmark in landmarks:
        for (x, y) in landmark:
            cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)  # Blue dots

    cv2.imshow('Video', frame)

def draw_detections(image, detections):
    """
    Draw bounding boxes and keypoints on the image.

    Args:
        image (np.ndarray): The image on which to draw.
        detections (list[dict]): Each dict contains 'bboxes' and 'keypoints' for detected objects.
    """
    for detection in detections:
        # Draw bounding boxes
        det = np.vstack(detection['bboxes'])
        scores = det[:, -1]
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        for bbox in det:
            # bbox = np.vstack(bbox)
            x1, y1, x2, y2, score = bbox
            if score > 0.5:  # Filter by confidence threshold
                # Draw the bounding box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Draw keypoints if available
        if 'keypoints' in detection:
            kp = np.vstack(detection['keypoints'])
            # kp = kp[order, :, :]
            for keypoint_set in detection['keypoints']:
                for x, y in keypoint_set:  # Each keypoint is a (x, y) tuple
                    # Draw each keypoint as a small circle
                    cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)

    return image


if __name__ == '__main__':
    main()
