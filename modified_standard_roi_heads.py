from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads
from detectron2.structures import Boxes, ImageList, Instances
from typing import Dict, List, Optional, Tuple
import torch

from modified_fast_rcnn_output_layers import ModifiedFastRCNNOutputLayers


class ModifiedStandardROIHeads(StandardROIHeads):
    def __init__(self, standard_roi_heads_instance: StandardROIHeads):
        mask_in_features = None
        mask_pooler = None
        mask_head = None
        keypoint_in_features = None
        keypoint_pooler = None
        keypoint_head = None

        if standard_roi_heads_instance.mask_on:
            mask_in_features = standard_roi_heads_instance.mask_in_features
            mask_pooler = standard_roi_heads_instance.mask_pooler
            mask_head = standard_roi_heads_instance.mask_head

        if standard_roi_heads_instance.keypoint_on:
            keypoint_in_features = standard_roi_heads_instance.keypoint_in_features
            keypoint_pooler = standard_roi_heads_instance.keypoint_pooler
            keypoint_head = standard_roi_heads_instance.keypoint_head

        super().__init__(box_in_features = standard_roi_heads_instance.box_in_features,
                         box_pooler = standard_roi_heads_instance.box_pooler,
                         box_head = standard_roi_heads_instance.box_head,
                         box_predictor = ModifiedFastRCNNOutputLayers(standard_roi_heads_instance.box_predictor),
                         mask_in_features = mask_in_features,
                         mask_pooler = mask_pooler,
                         mask_head = mask_head,
                         keypoint_in_features = keypoint_in_features,
                         keypoint_pooler = keypoint_pooler,
                         keypoint_head = keypoint_head,
                         train_on_pred_boxes = standard_roi_heads_instance.train_on_pred_boxes,
                         batch_size_per_image = standard_roi_heads_instance.batch_size_per_image,
                         positive_fraction = standard_roi_heads_instance.positive_fraction,
                         num_classes = standard_roi_heads_instance.num_classes,
                         proposal_matcher = standard_roi_heads_instance.proposal_matcher,
                         )


    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances