from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads

from modified_fast_rcnn_output_layers import ModifiedFastRCNNOutputLayers


class ModifiedStandardROIHeads(StandardROIHeads):
    def __init__(self, standard_roi_heads_instance: StandardROIHeads) -> None:
        super().__init__(box_in_features = standard_roi_heads_instance.box_in_features,
                         box_pooler = standard_roi_heads_instance.box_pooler,
                         box_head = standard_roi_heads_instance.box_head,
                         box_predictor = ModifiedFastRCNNOutputLayers(standard_roi_heads_instance.box_predictor),
                         mask_in_features = standard_roi_heads_instance.mask_in_features,
                         mask_pooler = standard_roi_heads_instance.mask_pooler,
                         mask_head = standard_roi_heads_instance.mask_head,
                         keypoint_in_features = standard_roi_heads_instance.keypoint_in_features,
                         keypoint_pooler = standard_roi_heads_instance.keypoint_pooler,
                         keypoint_head = standard_roi_heads_instance.keypoint_head,
                         train_on_pred_boxes = standard_roi_heads_instance.train_on_pred_boxes
                        )

