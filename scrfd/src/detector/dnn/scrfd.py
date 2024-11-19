import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from detector.dnn.single_stage import SingleStageDetector
from detector.utils.transforms import nms


class SCRFD(SingleStageDetector):

    def __init__(self,
                 backbone: nn.Module,
                 neck: nn.Module,
                 bbox_head: nn.Module,
                 train_cfg: dict = None,
                 test_cfg: dict = None,
                 pretrained=None,
                 use_kps: bool = True,
                 nms_thresh: float = 0.4):
        """
        Args:
            backbone:
            neck:
            bbox_head:
            train_cfg:
            test_cfg:
            pretrained:
        """
        super(SCRFD, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)
        self.use_kps = use_kps
        self.nms_thresh = nms_thresh

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_keypointss=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_keypointss, gt_bboxes_ignore)
        return losses

    def extract(self, img: Tensor, img_metas: list[dict], rescale=False) -> list[np.ndarray]:
        """Test function without test time augmentation.

        Args:
            img: torch.Tensor: Image
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        # Generate bounding boxes
        bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)

        # return bbox_list[0]
        # det_scale = img_metas[0]['det_scale']

        score_lst, bbox_lst, kp_lst = bbox_list[0]

        b, l = self.post_process(score_lst, bbox_lst, kp_lst, img)
        # print(res)
        bbox_data = {
            'bbox': b,
            'keypoints': l
        }
        return [bbox_data]

    def simple_test(self, img, img_metas: list[dict], rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        ## det scale
        im_ratio = float(img.shape[-1]) / img.shape[-2]
        model_ratio = 1.0
        if im_ratio > model_ratio:
            new_height = 640
            new_width = int(new_height / im_ratio)
        else:
            new_width = 640
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[-1]

        # Unpack the output
        cls_score, bbox_pred, kps_pred = outs if self.bbox_head.use_kps else (*outs, None)

        # Generate bounding boxes
        bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)

        mlvl_bboxes, mlvl_scores, mlvl_keypoints, score_lst, bbox_lst, kp_lst = bbox_list[0]
        b, l = self.post_process(score_lst, bbox_lst, kp_lst, img)
        # print(res)
        bbox_data = {
            'bbox': b,
            'keypoints': l
        }
        return [bbox_data]

    def feature_test(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def post_process(self, scores_list, bboxes_list, kps_list, image, max_num=0, metric='default'):
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list)
        if self.use_kps:
            kpss = np.vstack(kps_list)
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None

        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = image.shape[-1] // 2, image.shape[-2] // 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])

            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss
