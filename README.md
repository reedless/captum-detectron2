# captum-detectron2

After installing detectron2 library locally, edit the following lines to allow for Tensor input to faster rcnn

`detectron2/structures/image_list.py`

add .detach() to line 108
`pad_img.detach()[..., :img.shape[-2], :img.shape[-1]].copy_(img)`

`detectron2/modeing/meta_arch/rcnn.py`

remove dict access from line 224
`images = [x.to(self.device) for x in batched_inputs]`

change dict.get to just default values on line 239, 240
`height = image_size[0]` and `width = image_size[1]`