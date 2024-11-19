import cv2
import numpy as np
import torch
from torch import nn

from detector.backbones.mobilenet import MobileNetV1
from detector.dense_heads.scrfd_head import SCRFDHead
from detector.dnn.scrfd import SCRFD
from detector.necks.pafpn import PAFPN


class FaceDetector:
    def __init__(self, model_path: str = None, device: torch.device = torch.device('cpu')):
        self.model_path: str = model_path
        self.device: torch.device = device
        self.model: nn.Module = self.init_detector()

    def init_detector(self):
        """
        Args:
            model_path:
            device:

        Returns:

        """
        if self.model_path is not None:
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
                                  train_cfg=dict(
                                      assigner=dict(type='ATSSAssigner', topk=9),
                                      allowed_border=-1,
                                      pos_weight=-1,
                                      debug=False),
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
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True)  # Load on CPU
            state_dict = checkpoint.get('state_dict', checkpoint)  # Adjust if state_dict is nested
            model.load_state_dict(state_dict, strict=False)  # Use strict=False to ignore non-matching keys

            # checkpoint = load_checkpoint(model, model_path, map_location=map_loc)
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']

            # model.cfg = config  # save the config in the model for convenience
            model.to(self.device)
            model.eval()
            return model

    def __pre_process(self, img):
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
        return resized_img, cv2.dnn.blobFromImage(det_img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)

    def detect(self, img):
        # preprocess image
        resized_img, blob = self.__pre_process(img)

        # generate input to pass to dnn
        data = dict(img=[torch.from_numpy(blob).to(self.device)])
        data['img_metas'] = [[{'ori_shape': (360, 640, 3),
                               'img_shape': (360, 640, 3),
                               'pad_shape': (640, 640, 3),
                               'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32),
                               'batch_input_shape': (640, 640)}]]

        # forward the model
        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)[0]
        return resized_img, result
