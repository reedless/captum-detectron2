from typing import Dict, List

import torch
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess

from modified_image_list import ModifiedImageList

# TODO: fix class method for ModifiedImageList

class ModifiedGeneralizedRCNN(GeneralizedRCNN):
    def __init__(self, generalized_rcnn_instance):
        super().__init__()
        self.generalized_rcnn_instance = generalized_rcnn_instance

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x.to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ModifiedImageList.from_tensors(images, self.backbone.size_divisibility) # Extend ImageList to new object
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            # height = input_per_image.get("height", image_size[0])
            # width = input_per_image.get("width", image_size[1])            
            height = image_size[0]
            width = image_size[1]
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results