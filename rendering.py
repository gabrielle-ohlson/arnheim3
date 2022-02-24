#@title Imports:
import torch

# ------

# Invert image colours to have a white background?
INVERT_COLOURS = False

RENDER_EPSILON = 1e-8
RENDER_OVERLAP_TEMPERATURE = 0.1
RENDER_OVERLAP_ZERO_OFFSET = -5
RENDER_OVERLAP_MASK_THRESHOLD = 0.5
RENDER_TRANSPARENCY_MASK_THRESHOLD = 0.1

# ------

def population_render_transparency(x, b=None):
	"""Image rendering function that renders all patches on top of one another,
		 with transparency, using black as the transparent colour.

	Args:
		x: tensor of transformed RGB image patches of shape [S, B, 5, H, W].
		b: optional tensor of background RGB image of shape [S, 3, H, W].
	Returns:
		Tensor of rendered RGB images of shape [S, 3, H, W].
	"""
	# Sum the RGB patches [S, B, 3, H, W] as [S, 3, H, W].
	x = x[:, :, :3, :, :] * x[:, :, 3:4, :, :]
	y = x[:, :, :3, :, :].sum(1)
	if INVERT_COLOURS:
		y[:, :3, :, :] = 1.0 - y[:, :3, :, :]
	# Add backgrounds [S, 3, H, W].
	if b is not None:
		b = b.cuda() if x.is_cuda else b.cpu()
		y = torch.where(y.sum(1, keepdim=True) > RENDER_TRANSPARENCY_MASK_THRESHOLD,
										y[:, :3, :, :], b.unsqueeze(0)[:, :3, :, :])
	return y.clamp(0., 1.).permute(0, 2, 3, 1)


def population_render_masked_transparency(x, mode, b=None):
	"""Image rendering function that renders all patches on top of one another,
		 with transparency, using the alpha chanel as the mask colour.

	Args:
		x: tensor of transformed RGB image patches of shape [S, B, 5, H, W].
		mode: ["clipped"|"normed"] method of handling alpha with background.
		b: optional tensor of background RGB image of shape [S, 3, H, W].
	Returns:
		Tensor of rendered RGB images of shape [S, 3, H, W].
	"""
	# Get the patch mask [S, B, 1, H, W] and sum of masks [S, 1, H, W].
	mask = x[:, :, 3:4, :, :]
	mask_sum = mask.sum(1) + RENDER_EPSILON
	# Mask the RGB patches [S, B, 4, H, W] -> [S, B, 3, H, W].
	masked_x = x[:, :, :3, :, :] * mask
	# Compute mean of the RGB patches [S, B, 3, H, W] as [S, 3, H, W].
	x_sum = masked_x.sum(1)
	y = torch.where(
			mask_sum > RENDER_EPSILON, x_sum / mask_sum, mask_sum)
	# # Anti-aliasing on the countours of the sum of patches.
	# y = y * mask_sum.clamp(0., 1.) #TODO: #remove #? #*
	if INVERT_COLOURS:
		y[:, :3, :, :] = 1.0 - y[:, :3, :, :]
	# Add backgrounds [S, 3, H, W].
	if b is not None:
		b = b.cuda() if x.is_cuda else b.cpu()
		
		if mode == "normed":
			mask_max = mask_sum.max(
				dim=2, keepdim=True).values.max(dim=3, keepdim=True).values
			mask = mask_sum / mask_max
		elif mode == "clipped":
			# Anti-aliasing on the countours of the sum of patches.
			mask = mask_sum.clamp(0., 1.)
		else: raise ValueError(f"Unknown masked_transparency mode {mode}")

		y = y[:, :3, :, :] * mask + b.unsqueeze(0)[:, :3, :, :] * (1 - mask)

# 		y = torch.where(mask.sum(1) > RENDER_OVERLAP_MASK_THRESHOLD, y[:, :3, :, :],
# 									b.unsqueeze(0)[:, :3, :, :])
	return y.clamp(0., 1.).permute(0, 2, 3, 1)


def population_render_overlap(x, b=None, gamma=None):
	"""Image rendering function that overlays all patches on top of one another,
		 with semi-translucent overlap, using the alpha chanel as the mask colour
		 and the 5th channel as the order for the overlapped images.

	Args:
		x: tensor of transformed RGB image patches of shape [S, B, 5, H, W].
		b: optional tensor of background RGB image of shape [S, 3, H, W].
	Returns:
		Tensor of rendered RGB images of shape [S, 3, H, W].
	"""
	# Get the patch mask [S, B, 1, H, W].
	mask = x[:, :, 3:4, :, :]
	# Mask the patches [S, B, 4, H, W] -> [S, B, 3, H, W]
	masked_x = x[:, :, :3, :, :] * mask * mask
	# Mask the orders [S, B, 1, H, W] -> [S, B, 1, H, W]
	order = torch.where(
			mask > RENDER_OVERLAP_MASK_THRESHOLD,
			x[:, :, 4:, :, :] * mask / RENDER_OVERLAP_TEMPERATURE,
			mask + RENDER_OVERLAP_ZERO_OFFSET)
	# Get weights from orders [S, B, 1, H, W]
	weights = F.softmax(order, dim=1)
	# Apply weights to masked patches and compute mean over patches [S, 3, H, W].
	y = (weights * masked_x).sum(1)
	if INVERT_COLOURS:
		y[:, :3, :, :] = 1.0 - y[:, :3, :, :]
	if b is not None:
		b = b.cuda() if x.is_cuda else b.cpu()
		y = torch.where(mask.sum(1) > RENDER_OVERLAP_MASK_THRESHOLD, y[:, :3, :, :],
									b.unsqueeze(0)[:, :3, :, :])
	return y.clamp(0., 1.).permute(0, 2, 3, 1)
	
