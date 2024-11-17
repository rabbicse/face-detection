import argparse

import cv2
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter

import configs.scrfd.scrfd_500m_bnkps as conf
from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose


def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # prepare data from image: ndarray
    # data = dict(img=img)


    data = dict(img=[torch.from_numpy(img)])
    data['img_metas'] = [{'filename': None,
                          'ori_shape': (360, 640, 3),
                          'img_shape': (360, 640, 3),
                          'pad_shape': (640, 640, 3),
                          'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32),
                          'batch_input_shape': (640, 640)}]

    cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    # build the data pipeline
    # test_pipeline = Compose(cfg.data.test.pipeline)
    # data = test_pipeline(data)
    # data = collate([data], samples_per_gpu=1)
    # if next(model.parameters()).is_cuda:
    #     # scatter to specified GPU
    #     data = scatter(data, [device])[0]
    # else:
    #     for m in model.modules():
    #         assert not isinstance(
    #             m, RoIPool
    #         ), 'CPU inference with RoIPool is not supported currently.'
    #     # just get the actual data from DataContainer
    #     data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)[0]
    return result


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

    img = cv2.imread('messi-hair.jpg')
    im_ratio = float(img.shape[0]) / img.shape[1]
    model_ratio = 1.0
    if im_ratio > model_ratio:
        new_height = 640
        new_width = int(new_height / im_ratio)
    else:
        new_width = 640
        new_height = int(new_width * im_ratio)
    resized_img = cv2.resize(img, (new_width, new_height))
    det_img = np.zeros((640, 640, 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized_img

    input_size = tuple(det_img.shape[0:2][::-1])
    blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)

    # result = inference_detector(model, resized_img)
    result = inference_detector(model, blob)

    draw_img(resized_img, result['bbox'], result['keypoints'])


def draw_img(frame, boxes, landmarks):
    rect_color = (224, 128, 20)
    circle_color = (255, 0, 0)
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
            cv2.circle(frame, (int(x), int(y)), 4, circle_color, -1)  # Blue dots

    cv2.imshow('Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
