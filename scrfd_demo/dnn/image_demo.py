import argparse
import cv2
import numpy as np
import torch

from dnn.mobilenet import MobileNetV1
from dnn.pafpn import PAFPN
from dnn.scrfd import SCRFD
from dnn.scrfd_head import SCRFDHead


def init_detector(model_path: str = None, device: torch.device = torch.device('cpu')):
    """
    Args:
        model_path:
        device:

    Returns:

    """
    if model_path is not None:
        # construct backbone
        backbone = MobileNetV1(block_cfg=dict(stage_blocks=(2, 3, 2, 6),
                                              stage_planes=[16, 16, 40, 72, 152, 288]))

        # neck
        neck = PAFPN(in_channels=[40, 72, 152, 288],
                     out_channels=16,
                     start_level=1,
                     add_extra_convs='on_output',
                     num_outs=3)
        # construct bbox head
        bbox_head = SCRFDHead(num_classes=1,
                              in_channels=16,
                              stacked_convs=2,
                              feat_channels=64,
                              norm_cfg=dict(type='BN', requires_grad=True),
                              cls_reg_share=True,
                              strides_share=False,
                              dw_conv=True,
                              scale_mode=0,
                              anchor_generator=dict(
                                  type='AnchorGenerator',
                                  ratios=[1.0],
                                  scales=[1, 2],
                                  base_sizes=[16, 64, 256],
                                  strides=[8, 16, 32]),
                              loss_cls=dict(
                                  type='QualityFocalLoss',
                                  use_sigmoid=True,
                                  beta=2.0,
                                  loss_weight=1.0),
                              loss_dfl=False,
                              reg_max=8,
                              loss_bbox=dict(type='DIoULoss', loss_weight=2.0),
                              use_kps=True,
                              loss_kps=dict(
                                  type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=0.1),
                              # train_cfg=dict(
                              #     assigner=dict(type='ATSSAssigner', topk=9),
                              #     allowed_border=-1,
                              #     pos_weight=-1,
                              #     debug=False),
                              test_cfg=dict(
                                  nms_pre=-1,
                                  min_bbox_size=0,
                                  score_thr=0.02,
                                  nms=dict(type='nms', iou_threshold=0.45),
                                  max_per_img=-1)
                              )

        # initiate model
        model = SCRFD(backbone=backbone,
                      bbox_head=bbox_head,
                      neck=neck,
                      test_cfg=dict(
                          nms_pre=-1,
                          min_bbox_size=0,
                          score_thr=0.02,
                          nms=dict(type='nms', iou_threshold=0.45),
                          max_per_img=-1))

        # Load the state dict from the .pth file
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)  # Load on CPU
        state_dict = checkpoint.get('state_dict', checkpoint)  # Adjust if state_dict is nested
        model.load_state_dict(state_dict, strict=False)  # Use strict=False to ignore non-matching keys

        # checkpoint = load_checkpoint(model, model_path, map_location=map_loc)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        # else:
        #     model.CLASSES = get_classes('coco')

        # model.cfg = config  # save the config in the model for convenience
        model.to(device)
        model.eval()
        return model


def inference_detector(model, img, device):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    # cfg = model.cfg

    # prepare data from image: ndarray
    # data = dict(img=img)

    data = dict(img=[torch.from_numpy(img).to(device)])
    data['img_metas'] = [[{'filename': None,
                           'ori_shape': (360, 640, 3),
                           'img_shape': (360, 640, 3),
                           'pad_shape': (640, 640, 3),
                           'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32),
                           'batch_input_shape': (640, 640)}]]

    # cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)[0]
    return result


def parse_args():
    parser = argparse.ArgumentParser(description='Image Detection')
    parser.add_argument(
        '--checkpoint', type=str, default='../SCRFD_500M_KPS.pth', help='checkpoint file')
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
    print(args)

    # device
    device = torch.device(args.device)


    model = init_detector(args.checkpoint, device=device)

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
    result = inference_detector(model, blob, device=device)

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
