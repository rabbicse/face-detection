import numpy as np

from mmdet.core import bbox2result
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch


@DETECTORS.register_module()
class SCRFD(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SCRFD, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)
        self.use_kps = True
        self.nms_thresh = 0.4

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

    def simple_test(self, img, img_metas=None, rescale=False):
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
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        mlvl_bboxes, mlvl_scores, mlvl_keypoints, score_lst, bbox_lst, kp_lst = bbox_list[0]
        b, l = self.post_process(score_lst, bbox_lst, kp_lst, det_scale, img)
        # print(res)
        bbox_data = {
            'bbox': b,
            'keypoints': l
        }
        return [bbox_data]

        # Debugging: Check if keypoints are generated
        # kps_pred = np.vstack(kps_pred)
        # if kps_pred is not None:
        #     print("Keypoints Prediction (kps_pred) Shape:", kps_pred.shape)
        #     print("Keypoints Prediction (kps_pred) Sample:", kps_pred[0].detach().cpu().numpy())

        bbox_results = []
        for idx, (det_bboxes, det_labels, det_keypoints) in enumerate(bbox_list):
            bbox_data = {
                'bboxes': bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            }

            # # Include keypoints if available
            # if self.bbox_head.use_kps and kps_pred is not None:
            #     keypoints = kps_pred[idx].detach().cpu().numpy()
            #     # keypoints_list = [{'x': kp[0], 'y': kp[1]} for kp in keypoints]
            #     keypoints = keypoints.reshape((keypoints.shape[0], -1, 2))
            #     bbox_data['keypoints'] = keypoints

            # Include keypoints if keypoint prediction is enabled and available
            if self.bbox_head.use_kps and det_keypoints is not None:
                # Reshape keypoints to (N, num_keypoints, 2)
                det_keypoints = det_keypoints.detach().cpu().numpy()
                det_keypoints = det_keypoints.reshape((det_keypoints.shape[0], -1, 2))
                bbox_data['keypoints'] = det_keypoints

            bbox_results.append(bbox_data)
        print(bbox_results)
        return bbox_results

        # print(len(outs))
        # if torch.onnx.is_in_onnx_export():
        #     print('single_stage.py in-onnx-export')
        #     print(outs.__class__)
        #     cls_score, bbox_pred, kps_pred = outs
        #     for c in cls_score:
        #         print(c.shape)
        #     for c in bbox_pred:
        #         print(c.shape)
        #     #print(outs[0].shape, outs[1].shape)
        #     if self.bbox_head.use_kps:
        #         for c in kps_pred:
        #             print(c.shape)
        #         return (cls_score, bbox_pred, kps_pred)
        #     else:
        #         return (cls_score, bbox_pred)
        #     #return outs
        # bbox_list = self.bbox_head.get_bboxes(
        #     *outs, img_metas, rescale=rescale)
        # print(len(bbox_list))
        #
        # # skip post-processing when exporting to ONNX
        # #if torch.onnx.is_in_onnx_export():
        # #    return bbox_list
        #
        # bbox_results = [
        #     bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        #     for det_bboxes, det_labels in bbox_list
        # ]
        #
        # print(len(bbox_results))
        #
        # return bbox_results

    def feature_test(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def post_process(self, scores_list, bboxes_list, kps_list, det_scale, image, max_num=0, metric='default'):
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kps_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
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

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            index = np.where(ovr <= thresh)[0]
            order = order[index + 1]

        return keep
