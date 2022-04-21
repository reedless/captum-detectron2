import cv2
import torch
from captum.attr import (IntegratedGradients, GradientShap,
                         DeepLift, DeepLiftShap,
                         Saliency, InputXGradient,
                         Deconvolution, GuidedBackprop, GuidedGradCam,
                         FeatureAblation, FeaturePermutation, Occlusion,
                         NoiseTunnel)
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
import matplotlib.pyplot as plt
import numpy as np

# from modified_rcnn import ModifiedGeneralizedRCNN
from modified_fast_rcnn_output_layers import ModifiedFastRCNNOutputLayers
from modified_image_list import ModifiedImageList
from types import MethodType

img = cv2.imread('000000000001.jpg')
device = torch.device("cuda")
# device = torch.device("cpu")

# build and load faster rcnn model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

model = build_model(cfg).to(device).eval()
# DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

# modified = ModifiedGeneralizedRCNN(model).to(device).eval()

model.roi_heads.box_predictor = ModifiedFastRCNNOutputLayers(model.roi_heads.box_predictor)

def new_preprocess_image(self, batched_inputs: torch.Tensor):
      """
      Normalize, pad and batch the input images.
      """
      # print(type(batched_inputs))
      images = [x.to(self.device) for x in batched_inputs]
      images = [(x - self.pixel_mean) / self.pixel_std for x in images]
      images = ModifiedImageList.from_tensors(images, self.backbone.size_divisibility) # Extend ImageList to new object
      return images

def _new_postprocess(instances, batched_inputs: torch.Tensor, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        # print(type(batched_inputs))
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = image_size[0]
            width = image_size[1]
            from detectron2.modeling.postprocessing import detector_postprocess
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

model.preprocess_image = MethodType(new_preprocess_image, model)
model.__class__._postprocess = _new_postprocess
model.roi_heads.forward_with_given_boxes = MethodType(lambda self, x, y: y, model)

modified = model

DetectionCheckpointer(modified).load(cfg.MODEL.WEIGHTS)
modified.to(device)

print("Modified model loaded")

class WrapperModel(torch.nn.Module):
      def __init__(self):
            super().__init__()
            self.model = modified

      def forward(self, input):
            # just sum all the scores as per https://captum.ai/tutorials/Segmentation_Interpret
            outputs = self.model.inference(input, do_postprocess=False)
            acc = []
            for i in range(len(outputs)):
                  if outputs[i].shape[0] != 0:
                        acc.append(outputs[i].sum(dim=0).unsqueeze(0))
                  else:
                        acc.append(torch.cat([outputs[i],
                                          torch.zeros((1, outputs[i].shape[1])).to(device)]))
            return torch.cat(acc)

# define input and baseline
input_   = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device).type(torch.cuda.FloatTensor)
baseline = torch.zeros(input_.shape).to(device).type(torch.cuda.FloatTensor)
baseline_dist = torch.randn(5, 3, 480, 640).to(device) * 0.001

# run input through modified model to get number of instances
outputs = modified.inference(input_)

print(outputs[0]['instances'].pred_classes.unique())

modified.roi_heads.box_predictor.class_scores_only = True
wrapper_model = WrapperModel()

for pred_class in outputs[0]['instances'].pred_classes.unique():
      wrapper = WrapperModel()

      # Saliency
      saliency = Saliency(wrapper)
      attribution = saliency.attribute(input_, target=pred_class)

      attributions = attribution[0].permute(1,2,0).detach().cpu().numpy()
      attributions = np.sum(np.abs(attributions), axis=-1)

      print(np.sum(attributions), attributions.shape)

      fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 8))
      axs[0, 0].set_title('Attribution mask')
      axs[0, 0].imshow(attributions, cmap=plt.cm.inferno)
      axs[0, 0].axis('off')
      axs[0, 1].set_title('Overlay Saliency on Input image ')
      axs[0, 1].imshow(attributions, cmap=plt.cm.inferno)
      axs[0, 1].imshow(img, alpha=0.5)
      axs[0, 1].axis('off')
      plt.tight_layout()
      plt.savefig(f'Saliency_mask_{pred_class}.png', bbox_inches='tight') 

      # InputXGradient
      inputxgradient = InputXGradient(wrapper)
      attribution = inputxgradient.attribute(input_, target=pred_class)

      attributions = attribution[0].permute(1,2,0).detach().cpu().numpy()
      attributions = np.sum(np.abs(attributions), axis=-1)

      print(np.sum(attributions), attributions.shape)

      fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 8))
      axs[0, 0].set_title('Attribution mask')
      axs[0, 0].imshow(attributions, cmap=plt.cm.inferno)
      axs[0, 0].axis('off')
      axs[0, 1].set_title('Overlay InputXGradient on Input image ')
      axs[0, 1].imshow(attributions, cmap=plt.cm.inferno)
      axs[0, 1].imshow(img, alpha=0.5)
      axs[0, 1].axis('off')
      plt.tight_layout()
      plt.savefig(f'InputXGradient_mask_{pred_class}.png', bbox_inches='tight') 


