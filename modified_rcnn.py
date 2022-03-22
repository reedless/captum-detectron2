# from typing import Dict, List, Optional

# import torch
# from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
# from detectron2.modeling.postprocessing import detector_postprocess
# from detectron2.structures import Instances
# from detectron2.utils.events import get_event_storage

# from modified_image_list import ModifiedImageList
# from modified_standard_roi_heads import ModifiedStandardROIHeads


# class ModifiedGeneralizedRCNN(GeneralizedRCNN):
#     def __init__(self, generalized_rcnn_instance: GeneralizedRCNN):
#         super().__init__(backbone = generalized_rcnn_instance.backbone,
#                         proposal_generator = generalized_rcnn_instance.proposal_generator,
#                         roi_heads = ModifiedStandardROIHeads(generalized_rcnn_instance.roi_heads),
#                         pixel_mean = generalized_rcnn_instance.pixel_mean,
#                         pixel_std = generalized_rcnn_instance.pixel_std,
#                         input_format = generalized_rcnn_instance.input_format,
#                         vis_period = generalized_rcnn_instance.vis_period
#                         )
        

#     def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
#         """
#         Normalize, pad and batch the input images.
#         """
#         images = [x.to(self.device) for x in batched_inputs]
#         images = [(x - self.pixel_mean) / self.pixel_std for x in images]
#         images = ModifiedImageList.from_tensors(images, self.backbone.size_divisibility) # Extend ImageList to new object
#         return images


#     def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
#         """
#         Args:
#             batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
#                 Each item in the list contains the inputs for one image.
#                 For now, each item in the list is a dict that contains:

#                 * image: Tensor, image in (C, H, W) format.
#                 * instances (optional): groundtruth :class:`Instances`
#                 * proposals (optional): :class:`Instances`, precomputed proposals.

#                 Other information that's included in the original dicts, such as:

#                 * "height", "width" (int): the output resolution of the model, used in inference.
#                   See :meth:`postprocess` for details.

#         Returns:
#             list[dict]:
#                 Each dict is the output for one input image.
#                 The dict contains one key "instances" whose value is a :class:`Instances`.
#                 The :class:`Instances` object has the following keys:
#                 "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
#         """
#         if not self.training:
#             return self.inference(batched_inputs)

#         images = self.preprocess_image(batched_inputs)
#         # if "instances" in batched_inputs[0]:
#         #     gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
#         # else:
#         gt_instances = None

#         features = self.backbone(images.tensor)

#         if self.proposal_generator is not None:
#             proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
#         else:
#             assert "proposals" in batched_inputs[0]
#             proposals = [x["proposals"].to(self.device) for x in batched_inputs]
#             proposal_losses = {}

#         _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
#         if self.vis_period > 0:
#             storage = get_event_storage()
#             if storage.iter % self.vis_period == 0:
#                 self.visualize_training(batched_inputs, proposals)

#         losses = {}
#         losses.update(detector_losses)
#         losses.update(proposal_losses)
#         return losses

#     def inference(
#         self,
#         batched_inputs: List[Dict[str, torch.Tensor]],
#         detected_instances: Optional[List[Instances]] = None,
#         do_postprocess: bool = True,
#         class_scores_only: bool = False,
#     ):
#         """
#         Run inference on the given inputs.

#         Args:
#             batched_inputs (list[dict]): same as in :meth:`forward`
#             detected_instances (None or list[Instances]): if not None, it
#                 contains an `Instances` object per image. The `Instances`
#                 object contains "pred_boxes" and "pred_classes" which are
#                 known boxes in the image.
#                 The inference will then skip the detection of bounding boxes,
#                 and only predict other per-ROI outputs.
#             do_postprocess (bool): whether to apply post-processing on the outputs.

#         Returns:
#             When do_postprocess=True, same as in :meth:`forward`.
#             Otherwise, a list[Instances] containing raw network outputs.
#         """
#         assert not self.training

#         images = self.preprocess_image(batched_inputs)
#         features = self.backbone(images.tensor)

#         if detected_instances is None:
#             if self.proposal_generator is not None:
#                 proposals, _ = self.proposal_generator(images, features, None)
#             else:
#                 assert "proposals" in batched_inputs[0]
#                 proposals = [x["proposals"].to(self.device) for x in batched_inputs]

#             results, _ = self.roi_heads.forward(images, features, proposals, None, class_scores_only)
#         else:
#             detected_instances = [x.to(self.device) for x in detected_instances]
#             results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

#         if do_postprocess:
#             assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
#             return ModifiedGeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
#         else:
#             return results

#     @staticmethod
#     def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
#         """
#         Rescale the output instances to the target size.
#         """
#         # note: private function; subject to changes
#         processed_results = []
#         for results_per_image, input_per_image, image_size in zip(
#             instances, batched_inputs, image_sizes
#         ):
#             height = image_size[0]
#             width = image_size[1]
#             r = detector_postprocess(results_per_image, height, width)
#             processed_results.append({"instances": r})
#         return processed_results