import cv2
import torch
from captum.attr import (DeepLift, DeepLiftShap, GradientShap,
                         IntegratedGradients, LayerConductance,
                         NeuronConductance, NoiseTunnel)
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

img = cv2.imread('000000000001.jpg')

# build and load faster rcnn model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
model = build_model(cfg).eval()
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

# define input and baseline
input_   = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
baseline = torch.zeros(input_.shape)
# outputs = model(input_)


# Integrated Gradients
ig = IntegratedGradients(model)
attributions, delta = ig.attribute(input_, baseline, target=0, return_convergence_delta=True)
print('IG Attributions:', attributions)
print('Convergence Delta:', delta)


# Gradient SHAP
gs = GradientShap(model)

# We define a distribution of baselines and draw `n_samples` from that
# distribution in order to estimate the expectations of gradients across all baselines
baseline_dist = torch.randn(5, 3, 480, 640) * 0.001
attributions, delta = gs.attribute(input_, stdevs=0.09, n_samples=4, baselines=baseline_dist,
                                   target=0, return_convergence_delta=True)
print('GradientShap Attributions:', attributions)
print('Convergence Delta:', delta)
print('Average delta per example:', torch.mean(delta.reshape(input.shape[0], -1), dim=1))


# Deep Lift
dl = DeepLift(model)
attributions, delta = dl.attribute(input_, baseline, target=0, return_convergence_delta=True)
print('DeepLift Attributions:', attributions)
print('Convergence Delta:', delta)


# Deep Lift SHAP
dl = DeepLiftShap(model)
attributions, delta = dl.attribute(input_.float(), baseline_dist, target=0, return_convergence_delta=True)
print('DeepLiftSHAP Attributions:', attributions)
print('Convergence Delta:', delta)
print('Average delta per example:', torch.mean(delta.reshape(input.shape[0], -1), dim=1))


# Noise Tunnel + Integrated Gradients
ig = IntegratedGradients(model)
nt = NoiseTunnel(ig)
attributions, delta = nt.attribute(input_, nt_type='smoothgrad', stdevs=0.02, nt_samples=4,
      baselines=baseline, target=0, return_convergence_delta=True)
print('IG + SmoothGrad Attributions:', attributions)
print('Convergence Delta:', delta)
print('Average delta per example', torch.mean(delta.reshape(input.shape[0], -1), dim=1))


# Neuron Conductance
nc = NeuronConductance(model, model.backbone)
attributions = nc.attribute(input_, neuron_selector=1, target=0)
print('Neuron Attributions:', attributions)


# Layer Conductance
lc = LayerConductance(model, model.backbone)
attributions, delta = lc.attribute(input_, baselines=baseline, target=0, return_convergence_delta=True)
print('Layer Attributions:', attributions)
print('Convergence Delta:', delta)

