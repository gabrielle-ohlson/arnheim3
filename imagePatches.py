#@title Imports:
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
from skimage.transform import resize

# ------

#@title Functions used for loading patches

def add_binary_alpha_mask(patch):
  """Black pixels treated as having alpha=0, all other pixels have alpha=255"""
  shape = patch.shape
  mask = ((patch.sum(2) > 0) * 255).astype(np.uint8)
  return np.concatenate([patch, np.expand_dims(mask, -1)], axis=-1)


def resize_patch(patch, coeff):
  return resize(patch.astype(float),
                (int(np.round(patch.shape[0] * coeff)),
                 int(np.round(patch.shape[1] * coeff))))


def print_size_segmented_data(segmented_data):
  size_max = 0
  shape_max = None
  size_min = np.infty
  shape_min = None
  ws = []
  hs = []
  for i, segment in enumerate(segmented_data):
    segment = segment.swapaxes(0, 1) 
    shape_i = segment.shape
    size_i = shape_i[0] * shape_i[1]
    if size_i > size_max:
      shape_max = shape_i
      size_max = size_i
    if size_i < size_min:
      shape_min = shape_i
      size_min = size_i
    im_i = cv2.cvtColor(segment, cv2.COLOR_RGBA2BGRA)
    im_bgr = im_i[:, :, :3]
    im_mask = np.tile(im_i[:, :, 3:], (1, 1, 3))
    im_render = np.concatenate([im_bgr, im_mask], 1)
    print(f'Patch {i} of shape {shape_i}')
    cv2_imshow(im_render)
  print(f"{len(segmented_data)} patches, max {shape_max}, min {shape_min}")
