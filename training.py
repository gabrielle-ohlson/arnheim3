#@title Imports:
from matplotlib import pyplot as plt
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import random

#@title local imports:
from imageVideo import show_and_save

# ----
#@title Training settings

### Genetic algorithm methods
# **Microbial** - loser of randomly selected pair is replaced by mutated winner. A low selection pressure.
# **Evolutionary Strategies** - mutantions of the best individual replace the rest of the population. Much higher selection pressure than Microbial GA.
GA_METHOD = "Microbial"  #["Evolutionary Strategies", "Microbial"] #*training

# Gradient clipping during optimisation
GRADIENT_CLIPPING = 10.0

TRACE_EVERY = 500

# Limit the number of individuals shown during training
MAX_MULTIPLE_VISUALISATIONS = 7 #or 5 #?

# ----
#@title Image augmentation transformations for training

def augmentation_transforms(canvas_width, use_normalized_clip=False, use_augmentation=False):
	"""Image transforms to produce distorted crops to augment the evaluation.

	Args:
		canvas_width: width of the drawing canvas
		use_normalized_clip: Normalisation to better suit CLIP's training data
		use_augmentation: Image augmentation by affine transform

	Returns:
		transforms
	"""
	if use_normalized_clip and use_augmentation:
		augment_trans = transforms.Compose(
			[transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.6),
				transforms.RandomResizedCrop(canvas_width, scale=(0.7, 0.9)),
				transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
															(0.26862954, 0.26130258, 0.27577711))])
	elif use_augmentation:
		augment_trans = transforms.Compose([
			transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.6),
			transforms.RandomResizedCrop(canvas_width, scale=(0.7, 0.9)),
		])
	elif use_normalized_clip:
		augment_trans = transforms.Normalize(
			(0.48145466, 0.4578275, 0.40821073),
			(0.26862954, 0.26130258, 0.27577711))
	else:
		augment_trans = transforms.RandomPerspective(
			fill=1, p=0, distortion_scale=0)

	return augment_trans

# ------

#@title Training functions

def moving_average(a, n=3) :
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n


def plot_and_save_losses(loss_history, title="Losses", filename=None):
	losses = np.array(loss_history)
	plt.figure(figsize=(10,10))
	plt.xlabel("Training steps")
	plt.ylabel("Loss")
	plt.title(title)
	plt.plot(moving_average(losses, n=3))
	if filename:
		np.save(filename + ".npy", losses, allow_pickle=True)
		plt.savefig(filename + ".png")
	plt.show()


def make_optimizer(generator, learning_rate):
	"""Make optimizer for generator's parameters.

	Args:
		generator: generator model
		learning_rate: learning rate
		input_learing_rate: learning rate for input

	Returns:
		optimizer
	"""

	my_list = ['positions_top']
	params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, 
																							 generator.named_parameters()))))
	base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in 
																			my_list, generator.named_parameters()))))
	lr_scheduler = torch.optim.SGD([{'params': base_params},
																	{'params': params, 'lr': learning_rate}],
																		lr=learning_rate)
	return lr_scheduler


def create_augmented_batch(images, augment_trans, text_features, USE_NORMALIZED_CLIP=True): # USE_NORMALIZED_CLIP is #new
	"""Create batch of images to be evaluated.
	
	Returns:
		img_batch: For compositional images the batch contains all the regions.
				Otherwise the batch contains augmented versions of the original images.
		num_augs: number of images per original image
		expanded_text_features: a text feature for each augmentation
		loss_weights: weights for loss associated with each augmentation image
	"""
	images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
	expanded_text_features = []

	USE_IMAGE_AUGMENTATIONS = False #temp
	NUM_AUGS = 4 #temp

	if USE_IMAGE_AUGMENTATIONS:
		num_augs = NUM_AUGS
		img_augs = [] 
		for n in range(NUM_AUGS):
			img_n = augment_trans(images)
			img_augs.append(img_n)
			expanded_text_features.append(text_features[0])
		img_batch = torch.cat(img_augs)
		# Given images [P0, P1] and augmentations [a0(), a1()], output format:
		# [a0(P0), a0(P1), a1(P0), a1(P1)]
	else:
		num_augs = 1
		if USE_NORMALIZED_CLIP:
			img_batch = augment_trans(images)
		else:
			img_batch = images
		expanded_text_features.append(text_features[0])
	return img_batch, num_augs, expanded_text_features, [1] * NUM_AUGS


def create_compositional_batch(images, augment_trans, text_features):
	"""Create 10 sub-images per image by augmenting each with 3x3 crops.

	Args:
		images: population of N images, format [N, C, H, W]

	Returns:
		Tensor of all compositional sub-images + originals; [N*10, C, H, W] format:
				[x0_y0(P0) ... x0_y0(PN), ..., x2_y2(P0) ... x2_y2(PN), P0, ..., PN]
		10: Number of sub-images + whole, per original image.
		expanded_text_features: list of text features, 1 for each composition image
		loss_weights: weights for the losses corresponding to each composition image
		"""
	if len(text_features) != 10:
		# text_features should already be 10 in size.
		raise ValueError(
				"10 text prompts required for compositional image creation")
	resize_for_clip = transforms.Compose([transforms.Scale((224,224))])
	img_swap = torch.swapaxes(images, 3, 1)
	ims = []
	i = 0
	for x in range(3):
		for y in range(3):
			#  print(f"Prompt for x={x}, y={y} is {PROMPTS[i]}")
			for k in range(images.shape[0]):
				ims.append(resize_for_clip(
						img_swap[k][:, y * 112 : y * 112 + 224, x * 112 : x * 112 + 224]))  
			i += 1
	
	# Top-level (whole) images
	for k in range(images.shape[0]):
		ims.append(resize_for_clip(img_swap[k]))
	all_img = torch.stack(ims)
	all_img = torch.swapaxes(all_img, 1, 3)
	all_img = all_img.permute(0, 3, 1, 2)  # NHWC -> NCHW
	all_img = augment_trans(all_img)

	# Last image gets 9 times as much weight
	common_weight = 1 / 5
	loss_weights = [common_weight] * 9
	loss_weights.append(9 * common_weight)
	return all_img, 10, text_features, loss_weights

# ------

#@title Evaluation and step of optimization

# Show each image being evaluated for debugging purposes.
# VISUALISE_BATCH_IMAGES = False

def evaluation(device, dir_results, optim_steps, t, clip_enc, generator, augment_trans, text_features, compositional_image, prompts, VISUALISE_BATCH_IMAGES = False): #device is #new #dir_results is #new # optim_steps is #new
	"""Do a step of evaluation, returning images and losses.

	Args:
		t: step count
		clip_enc: model for CLIP encoding
		generator: drawing generator to optimise
		augment_trans: transforms for image augmentation
		text_features: tuple with the prompt two negative prompts
		compositional_image: use multiple CLIPs and prompts
		prompts: for debugging/visualisation - the list of text prompts
	
	Returns:
		loss: torch.Tensor of single combines loss
		losses_separate_np: numpy array of loss for each image
		losses_individuals_np: numpy array with loss for each population individual
		img_np: numpy array of images from the generator
	"""

	# Annealing parameters.
	params = {'gamma': t / optim_steps}

	# Rebuild the generator.
	img = generator(params)
	img_np = img.detach().cpu().numpy()

	# Create images for different regions
	pop_size = img.shape[0]
	if compositional_image:
		(img_batch, num_augs, text_features, loss_weights
		 ) = create_compositional_batch(img, augment_trans, text_features)
	else:
		(img_batch, num_augs, text_features, loss_weights
		 ) = create_augmented_batch(img, augment_trans, text_features)
	losses = torch.zeros(pop_size, num_augs).to(device)

	# Compute and add losses after augmenting the image with transforms.
	img_batch = torch.clip(img_batch, 0, 1)  # clip the images.
	image_features = clip_enc.encode_image(img_batch)
	count = 0
	for n in range(num_augs):  # number of augmentations or composition images
		for p in range(pop_size):
			loss = torch.cosine_similarity(
					text_features[n], image_features[count:count+1], dim=1
					)[0] * loss_weights[n]
			losses[p, n] -= loss
			if VISUALISE_BATCH_IMAGES and t % 500 == 0:
				# Show all the images in the batch along with their losses.
				if compositional_image:
					print(f"Loss {loss} for image region with prompt {prompts[n]}:")
				else:
					print(f"Loss {loss} for image augmentation with prompt {prompts[0]}:")
				show_and_save(img_batch[count].unsqueeze(0), dir_results, img_format="SCHW")
			count += 1
	loss = torch.sum(losses) / pop_size
	losses_separate_np = losses.detach().cpu().numpy()
	# Sum losses for all each population individual.
	losses_individuals_np = losses_separate_np.sum(axis=1)
	return loss, losses_separate_np, losses_individuals_np, img_np


def step_optimization(device, dir_results, optim_steps, t, clip_enc, lr_scheduler, generator, augment_trans, text_features, compositional_image, prompts, final_step=False): #device is #new #dir_results is #new # optim_steps is #new
	"""Do a step of optimization.

	Args:
		t: step count
		clip_enc: model for CLIP encoding
		lr_scheduler: optimizer
		generator: drawing generator to optimise
		augment_trans: transforms for image augmentation
		text_features: list or 1 or 9 prompts for normal and compositional creation
		compositional_image: use multiple CLIPs and prompts
		prompts: for debugging/visualisation - the list of text prompts
		final_step: if True does extras such as saving the model

	Returns:
		losses_np: numpy array with loss for each population individual
		losses_separate_np: numpy array of loss for each image
	"""

	# Anneal learning rate and other parameters.
	if t == int(optim_steps / 3):
		for g in lr_scheduler.param_groups:
			g["lr"] = g["lr"] / 2.0
	if t == int(optim_steps * (2/3)):
		for g in lr_scheduler.param_groups:
			g["lr"] = g["lr"] / 2.0
	params = {'gamma': t / optim_steps}

	# Forward pass. 
	lr_scheduler.zero_grad()
	loss, losses_separate_np, losses_np, img_np = evaluation(
			device, dir_results, optim_steps,
			t, clip_enc, generator, augment_trans, text_features, compositional_image,
			prompts)

	# Backpropagate the gradients.
	loss.backward()
	torch.nn.utils.clip_grad_norm(generator.parameters(), GRADIENT_CLIPPING)
	
	# Decay the learning rate.
	lr_scheduler.step()

	# Render the big version.
	if final_step:
		show_and_save(img_np, dir_results, t=t, img_format="SHWC")
		print("Saving model...")
		torch.save(generator.state_dict(), os.path.join(dir_results, "/generator.pt"))
		# torch.save(generator.state_dict(), pathlib.PurePath(DIR_RESULTS, "/generator.pt"))
		# torch.save(generator.state_dict(), DIR_RESULTS + "/generator.pt")

	if t % TRACE_EVERY == 0:
		show_and_save(img_np, dir_results,
									max_display=MAX_MULTIPLE_VISUALISATIONS,
									stitch=True, img_format="SHWC")

		print("Iteration {:3d}, rendering loss {:.6f}".format(t, loss.item()))
	return losses_np, losses_separate_np, img_np

# ------

#@title Evolution functions

def population_evolution_step(generator, losses):
	"""GA for the population."""
	if GA_METHOD == "Microbial":
		# Competition between 2 random individuals; mutated winner replaces loser.
		indices = list(range(len(losses)))
		random.shuffle(indices)
		select_1, select_2 = indices[0], indices[1]
		if losses[select_1] < losses[select_2]:
			# print(f"Replacing {select_2} with {select_1}")
			generator.copy_and_mutate_s(select_1, select_2)
		else:
			# print(f"Replacing {select_1} with {select_2}")
			generator.copy_and_mutate_s(select_2, select_1)
	elif GA_METHOD == "Evolutionary Strategies":
		# Replace rest of population with mutants of the best."""
		winner = np.argmin(losses)
		for other in range(len(losses)):
			if other == winner:
				continue
			generator.copy_and_mutate_s(winner, other)
