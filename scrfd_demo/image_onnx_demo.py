import argparse
import cv2

from face_detector import FaceDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_id', default=-1, help='image file path')
    parser.add_argument('--model', default='model_1_kps.onnx', help='model file path')
    args = parser.parse_args()

    detector = FaceDetector(onnx_file=args.model)
    frame = cv2.imread('./demo/messi-hair.jpg')

    boxes, landmarks = detector.detect(frame, input_size=(640, 640))
    boxes = boxes.astype('int32')

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

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
