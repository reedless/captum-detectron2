import cv2
import torch
from captum.attr import (DeepLift, DeepLiftShap, GradientShap,
                         IntegratedGradients, LayerConductance,
                         NeuronConductance, NoiseTunnel, LayerGradientXActivation)
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

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
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

model = build_model(cfg).to(device).eval()
# DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

# modified = ModifiedGeneralizedRCNN(model).to(device).eval()

model.roi_heads.box_predictor = ModifiedFastRCNNOutputLayers(model.roi_heads.box_predictor)

def new_preprocess_image(self, batched_inputs: torch.Tensor):
      """
      Normalize, pad and batch the input images.
      """
      print(type(batched_inputs))
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
        print(type(batched_inputs))
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
            print(len(outputs), outputs)
            for i in range(len(outputs)):
                  print('lll: ', outputs[i])
                  print('hiii', outputs[i].sum(dim=0).unsqueeze(0).shape)
                  if outputs[i].shape[0] != 0:
                        return outputs[i].sum(dim=0).unsqueeze(0)

# define input and baseline
input_   = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
baseline = torch.zeros(input_.shape).to(device)
baseline_dist = torch.randn(5, 3, 480, 640).to(device) * 0.001

# run input through modified model to get number of instances
outputs = modified.inference(input_)

print(outputs[0]['instances'].pred_classes.unique())

modified.roi_heads.box_predictor.class_scores_only = True
wrapper_model = WrapperModel()

for pred_class in outputs[0]['instances'].pred_classes.unique():
      # print(("Selecting instance prediction of "
      #       "class {} with "
      #       "score {} probability.".format(outputs[0]['instances'].pred_classes[i], outputs[0]['instances'].scores[i])
      #       ))

      # # LayerGradientXActivation
      # lg = LayerGradientXActivation(wrapper_model, wrapper_model.model.backbone) 
      # attributions = lg.attribute(input_, target=pred_class, attribute_to_layer_input=True)
      # print('LayerGradientXActivation Attributions:', attributions)

      # Integrated Gradients
      wrapper = WrapperModel()
      ig = IntegratedGradients(wrapper)
      attributions, delta = ig.attribute(input_, 
                                         target=pred_class, 
                                    #      additional_forward_args = (outputs[0]['instances'][0].pred_classes[i], 
                                    #                                 len(outputs[0]['instances'].class_scores[0])),
                                         return_convergence_delta=True)
      print('Convergence Delta:', delta)


      # # Gradient SHAP
      # gs = GradientShap(wrapper)

      # # We define a distribution of baselines and draw `n_samples` from that
      # # distribution in order to estimate the expectations of gradients across all baselines
      # attributions, delta = gs.attribute(input_, stdevs=0.09, n_samples=4, baselines=baseline_dist,
      #                               target=pred_class, return_convergence_delta=True)
      # print('GradientShap Attributions:', attributions)
      # print('Convergence Delta:', delta)
      # print('Average delta per example:', torch.mean(delta.reshape(input.shape[0], -1), dim=1))


      # # Deep Lift
      # dl = DeepLift(wrapper)
      # attributions, delta = dl.attribute(input_, baseline, target=pred_class, return_convergence_delta=True)
      # print('DeepLift Attributions:', attributions)
      # print('Convergence Delta:', delta)


      # # Deep Lift SHAP
      # dl = DeepLiftShap(wrapper)
      # attributions, delta = dl.attribute(input_.float(), baseline_dist, target=0, return_convergence_delta=True)
      # print('DeepLiftSHAP Attributions:', attributions)
      # print('Convergence Delta:', delta)
      # print('Average delta per example:', torch.mean(delta.reshape(input.shape[0], -1), dim=1))


      # # Noise Tunnel + Integrated Gradients
      # ig = IntegratedGradients(wrapper)
      # nt = NoiseTunnel(ig)
      # attributions, delta = nt.attribute(input_, nt_type='smoothgrad', stdevs=0.02, nt_samples=4,
      #       baselines=baseline, target=0, return_convergence_delta=True)
      # print('IG + SmoothGrad Attributions:', attributions)
      # print('Convergence Delta:', delta)
      # print('Average delta per example', torch.mean(delta.reshape(input.shape[0], -1), dim=1))


      # # Neuron Conductance
      # nc = NeuronConductance(wrapper, model.backbone)
      # attributions = nc.attribute(input_, neuron_selector=1, target=0)
      # print('Neuron Attributions:', attributions)


      # # Layer Conductance
      # lc = LayerConductance(wrapper, model.backbone)
      # attributions, delta = lc.attribute(input_, baselines=baseline, target=0, return_convergence_delta=True)
      # print('Layer Attributions:', attributions)
      # print('Convergence Delta:', delta)

