#TODO

#@title Imports:
import clip
import copy
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
from skimage.transform import resize
import time
import torch

### local imports
from imageVideo import layout_img_batch, show_and_save, VideoWriter

from popTransforms import PopulationAffineTransforms, PopulationOrderOnlyTransforms, PopulationColourHSVTransforms, PopulationColourRGBTransforms

from rendering import population_render_transparency, population_render_masked_transparency, population_render_overlap

from training import augmentation_transforms, plot_and_save_losses, make_optimizer, evaluation, step_optimization, population_evolution_step

# ------ #TODO: make these a dict

collage_settings = {
	'NUM_PATCHES': 100,
	'CANVAS_WIDTH': 224,
	'CANVAS_HEIGHT': 224,
	'HIGH_RES_MULTIPLIER': 4,
	# Number of training steps
	'OPTIM_STEPS': 200, # (min:200, max:20000)

	'LEARNING_RATE': 0.1, # (min:0.0, max:0.6)
	# Render methods
	# **opacity** patches overlay each other using a combination of alpha and depth,
	# **transparency** _adds_ patch colours (black therefore appearing transparent),
	# and **masked transparency** blends patches using the alpha channel.
	'RENDER_METHOD': "transparency", # ["opacity", "transparency", "masked_transparency"]
	
	'INITIAL_MIN_RGB': 0.1,
	'INITIAL_MAX_RGB': 0.5
# * INITIAL_MIN_RGB=0.7; INITIAL_MAX_RGB=1.0
}

#@title evolution settings

# For evolution set POP_SIZE greater than 1 
POP_SIZE = 2 # (min:1, max:100} #*collage
EVOLUTION_FREQUENCY = 100

PATCH_MUTATION_PROBABILITY = 1 # (min:0.0, max:1.0) #*collage
USE_EVOLUTION = POP_SIZE > 1

# ------

#@title General settings
# NUM_PATCHES = 100

# ------

#@title Collage configuration settings

# # Render methods
# # **opacity** patches overlay each other using a combination of alpha and depth,
# # **transparency** _adds_ patch colours (black therefore appearing transparent),
# # and **masked transparency** blends patches using the alpha channel.
# RENDER_METHOD = "transparency" # ["opacity", "transparency", "masked_transparency"]
# NUM_PATCHES = 100
COLOUR_TRANSFORMATIONS = "RGB space" # ["none", "RGB space", "HSV space"]


# CANVAS_WIDTH = 224
# CANVAS_HEIGHT = 224
# MULTIPLIER_BIG_IMAGE = 4

# ------
#@title Training settings

# Number of training steps
# OPTIM_STEPS = 200 # (min:200, max:20000)

# LEARNING_RATE = 0.1 # (min:0.0, max:0.6)

USE_IMAGE_AUGMENTATIONS = True

# Normalize colours for CLIP, generally leave this as True
USE_NORMALIZED_CLIP = True

# Initial random search size (1 means no search)
INITIAL_SEARCH_SIZE = 1 #(min:1, max:50)

# ------
#@title misc functions:

def text_features(prompts, device, clip_model):
	# Compute CLIP features for all prompts.
	text_inputs = []
	for prompt in prompts:
		text_inputs.append(clip.tokenize(prompt).to(device))

	features = []
	with torch.no_grad():
		for text_input in text_inputs:
			features.append(clip_model.encode_text(text_input))
	return features

# ------
#@title Collage network definition

class PopulationCollage(torch.nn.Module):
	"""Population-based segmentation collage network.
	
	Image structure in this class is SCHW."""
	def __init__(self, device, pop_size=1, is_high_res=False, segmented_data=None, background_image=None, settings=collage_settings): #device is #new
		"""Constructor, relying on global parameters."""
		super(PopulationCollage, self).__init__()
						
		self._device = device

		self._settings = settings
		
		self._CANVAS_WIDTH = self._settings['CANVAS_WIDTH']
		self._CANVAS_HEIGHT = self._settings['CANVAS_HEIGHT']
		self._HIGH_RES_MULTIPLIER = self._settings['HIGH_RES_MULTIPLIER']

		# Population size.
		self._pop_size = pop_size

		# Create the spatial transformer and colour transformer for patches.
		self.spatial_transformer = PopulationAffineTransforms(
			self._device, num_patches=self._settings['NUM_PATCHES'], pop_size=pop_size).cuda()
		if COLOUR_TRANSFORMATIONS == "HSV space":
			self.colour_transformer = PopulationColourHSVTransforms(
				self._device, num_patches=self._settings['NUM_PATCHES'], pop_size=pop_size).cuda()
		elif COLOUR_TRANSFORMATIONS == "RGB space":
			self.colour_transformer = PopulationColourRGBTransforms(
				self._device, num_patches=self._settings['NUM_PATCHES'], pop_size=pop_size,
				settings={'INITIAL_MIN_RGB': self._settings['INITIAL_MIN_RGB'], 'INITIAL_MAX_RGB': self._settings['INITIAL_MAX_RGB']}
			).cuda()
		else:
			self.colour_transformer = PopulationOrderOnlyTransforms(
				self._device, num_patches=self._settings['NUM_PATCHES'], pop_size=pop_size).cuda()

		# Optimisation is run in low-res, final rendering is in high-res.
		self._high_res = is_high_res

		# Store the background image (low- and high-res).
		self._background_image = background_image
		if self._background_image is not None:
			print(f'Background image of size {self._background_image.shape}')

		# Store the dataset (low- and high-res).
		self._dataset = segmented_data
		#print(f'There are {len(self._dataset)} image patches in the dataset')

		# Initial set of indices, pointing to the NUM_PATCHES first dataset images. 
		self.patch_indices = [np.arange(self._settings['NUM_PATCHES']) % len(self._dataset)
													for _ in range(pop_size)]

		# Patches in low and high-res.
		self.patches = None
		self.store_patches()

	def store_patches(self, population_idx=None):
		device = self._device

		"""Store the image patches for each population element."""
		t0 = time.time()

		if population_idx is not None and self.patches is not None:
			list_indices = [population_idx]
			#print(f'Reload {NUM_PATCHES} image patches for [{population_idx}]')
			self.patches[population_idx, :, :4, :, :] = 0
		else:
			list_indices = np.arange(self._pop_size)
			#print(f'Store {NUM_PATCHES} image patches for [1, ..., {self._pop_size}]')
			if self._high_res:
				self.patches = torch.zeros(
					1, self._settings['NUM_PATCHES'], 5, self._CANVAS_HEIGHT * self._HIGH_RES_MULTIPLIER,
					self._CANVAS_WIDTH * self._HIGH_RES_MULTIPLIER).to('cpu')
			else:
				self.patches = torch.zeros(
					self._pop_size, self._settings['NUM_PATCHES'], 5, self._CANVAS_HEIGHT, self._CANVAS_WIDTH
					).to(device)
			self.patches[:, :, 4, :, :] = 1.0

		# Put the segmented data into the patches.
		for i in list_indices:
			for j in range(self._settings['NUM_PATCHES']):
				k = self.patch_indices[i][j]
				patch_j = torch.tensor(
					self._dataset[k].swapaxes(0, 2) / 255.0).to(device)
				width_j = patch_j.shape[1]
				height_j = patch_j.shape[2]
				if self._high_res:
					w0 = int((self._CANVAS_WIDTH * self._HIGH_RES_MULTIPLIER - width_j) / 2.0)
					h0 = int((self._CANVAS_HEIGHT * self._HIGH_RES_MULTIPLIER - height_j) / 2.0)
				else:
					w0 = int((self._CANVAS_WIDTH - width_j) / 2.0)
					h0 = int((self._CANVAS_HEIGHT - height_j) / 2.0)
				if w0 < 0 or h0 < 0:
					import pdb; pdb.set_trace()
				self.patches[i, j, :4, w0:(w0 + width_j), h0:(h0 + height_j)] = patch_j
		t1 = time.time()
		#print('Updated patches in {:.3f}s'.format(t1-t0))

	def copy_and_mutate_s(self, parent, child):
		with torch.no_grad():
			# Copy the patches indices from the parent to the child.
			self.patch_indices[child] = copy.deepcopy(self.patch_indices[parent])
			
			# Mutate the child patches with a single swap from the original dataset. 
			if PATCH_MUTATION_PROBABILITY > np.random.uniform():
				idx_dataset  = np.random.randint(len(self._dataset))
				idx_patch  = np.random.randint(self._settings['NUM_PATCHES'])
				self.patch_indices[child][idx_patch] = idx_dataset

			# Update all the patches for the child.
			self.store_patches(child)
	
			self.spatial_transformer.copy_and_mutate_s(parent, child)
			self.colour_transformer.copy_and_mutate_s(parent, child)

	def copy_from(self, other, idx_to, idx_from):
		"""Copy parameters from other collage generator, for selected indices."""
		assert idx_to < self._pop_size
		with torch.no_grad():
			self.patch_indices[idx_to] = copy.deepcopy(other.patch_indices[idx_from])
			self.store_patches(idx_to)
			self.spatial_transformer.copy_from(
				other.spatial_transformer, idx_to, idx_from)
			self.colour_transformer.copy_from(
				other.colour_transformer, idx_to, idx_from)

	def forward(self, params=None):
		"""Input-less forward function."""

		shifted_patches = self.spatial_transformer(self.patches)
		background_image = self._background_image
		coloured_patches = self.colour_transformer(shifted_patches)
		if self._settings['RENDER_METHOD'] == "transparency":
			img = population_render_transparency(coloured_patches, background_image)
		elif self._settings['RENDER_METHOD'] == "masked_transparency":
			img = population_render_masked_transparency(
				coloured_patches, background_image)
		elif self._settings['RENDER_METHOD'] == "opacity":
			if params is not None and 'gamma' in params:
				gamma = params['gamma']
			else:
				gamma = None
			img = population_render_overlap(
				coloured_patches, background_image)
		else:
			print("Unhandled render method")
		return img

	def tensors_to(self, device):
		self.spatial_transformer.tensor_to(device)
		self.colour_transformer.tensor_to(device)
		self.patches = self.patches.to(device)

# ------

#@title CollageMaker class

class CollageMaker():
	def __init__(self, device, clip_model, dir_results, prompts, segmented_data, background_image, compositional_image, output_dir, file_basename, video_steps, population_video, settings): #device is #new #clip_model is #new #dir_results is #new
		self._device = device
		self._clip_model = clip_model
		self._dir_results = dir_results #new

		self._settings = settings
		self._CANVAS_WIDTH = self._settings['CANVAS_WIDTH']
		self._CANVAS_HEIGHT = self._settings['CANVAS_HEIGHT']
				
		self._prompts = prompts
		self._segmented_data = segmented_data
		self._background_image = background_image
		self._compositional_image = compositional_image
		self._file_basename = file_basename
		self._output_dir = output_dir
		self._population_video = population_video

		self._video_steps = video_steps
		if self._video_steps:
			self._video_writer = VideoWriter(
				filename=f"{self._output_dir}/{self._file_basename}.mp4")
			if self._population_video:
				self._population_video_writer = VideoWriter(
					filename=f"{self._output_dir}/{self._file_basename}_pop_sample.mp4")
		
		if self._compositional_image:
			if len(self._prompts) != 10:
				raise ValueError(
					"Missing compositional image prompts; found {len(self._prompts)}")
			print("Global prompt is", self._prompts[-1])
			print("Composition prompts", self._prompts)
		else:
			if len(self._prompts) != 1:
				raise ValueError(
						"Missing compositional image prompts; found {len(self._prompts)}")
			print("CLIP prompt", self._prompts[0])
		
		# Prompt to CLIP features.
		self._prompt_features = text_features(self._prompts, self._device, self._clip_model)
		self._augmentations = augmentation_transforms(
			224,
			use_normalized_clip=USE_NORMALIZED_CLIP,
			use_augmentation=USE_IMAGE_AUGMENTATIONS)
		
		# Create population of collage generators.
		self._generator = PopulationCollage(
			device=self._device,
			is_high_res=False,
			pop_size=POP_SIZE,
			segmented_data=segmented_data,
			background_image=background_image,
			settings=self._settings
		)
		
		# Initial search over hyper-parameters.
		if INITIAL_SEARCH_SIZE > 1:
			print(f'\nInitial random search over {INITIAL_SEARCH_SIZE} individuals')
			for j in range(POP_SIZE):
				generator_search = PopulationCollage(
					device=self._device,
					is_high_res=False,
					pop_size=INITIAL_SEARCH_SIZE,
					segmented_data=segmented_data,
					background_image=background_image,
					settings=self._settings
				)
					# compositional_image=self._compositional_image)
				_, _, losses, _ = evaluation(
					self._device, self._dir_results, self._settings['OPTIM_STEPS'], 0, self._clip_model, generator_search, self._augmentations, self._prompt_features, self._compositional_image, self._prompts) #new #?
					# 0, self._clip_model, generator_search, augmentations, prompt_features) #go back! #?
				print(f"Search {losses}")
				idx_best = np.argmin(losses)
				self._generator.copy_from(generator_search, j, idx_best) #new #?
				# generator.copy_from(generator_search, j, idx_best) #go back! #?
				del generator_search
			print(f'Initial random search done\n')
		
		self._optimizer = make_optimizer(self._generator, self._settings['LEARNING_RATE'])
		self._step = 0
		self._losses_history = []
		self._losses_separated_history = []

	@property
	def generator(self):
		return self._generator
		
	@property
	def step(self):
		return self._step

	def loop(self):
		"""Main optimisation/image generation loop. Can be interrupted."""
		OPTIM_STEPS = self._settings['OPTIM_STEPS']

		if self._step == 0:
			print(f'Starting optimization of collage. ({OPTIM_STEPS} steps total.)')
		else:
			print(f'Continuing optimization of collage at step {self._step}.') #?
			if self._video_steps:
				print(f"Aborting video creation (does not work when interrupted).")
				self._video_steps = 0
				self._video_writer = None
				if self._population_video_writer:
					self._population_video_writer = None
		
		while self._step < OPTIM_STEPS:
			print(f'Continuing optimization of collage at step ({self._step} / {OPTIM_STEPS}).') #new #?

			last_step = self._step == (OPTIM_STEPS - 1)
			losses, losses_separated, img_batch = step_optimization(
				self._device, self._dir_results, OPTIM_STEPS, self._step, self._clip_model, self._optimizer, self._generator,
				self._augmentations, self._prompt_features, 
				self._compositional_image, self._prompts, final_step=last_step)
			self._add_video_frames(img_batch, losses)
			self._losses_history.append(losses)
			self._losses_separated_history.append(losses_separated)
		
			if USE_EVOLUTION and self._step and self._step % EVOLUTION_FREQUENCY == 0:
				population_evolution_step(self._generator, losses) 
			self._step += 1
		

	def high_res_render(self, 
											segmented_data_high_res, 
											background_image_high_res,
											gamma=1.0, 
											show=True,
											save=True):
		"""Save and/or show a high res render using high-res patches."""
		generator = PopulationCollage(
			device=self._device,
			is_high_res=True,
			pop_size=1,
			segmented_data=segmented_data_high_res,
			background_image=background_image_high_res,
			settings=self._settings
		)
		idx_best = np.argmin(self._losses_history[-1])
		print(f'Lowest loss for indices: {idx_best}')
		generator.copy_from(self._generator, 0, idx_best)
		# Show high res version given a generator
		generator_cpu = copy.deepcopy(generator)
		generator_cpu = generator_cpu.to('cpu')
		generator_cpu.tensors_to('cpu')
	
		params = {'gamma': gamma}
		with torch.no_grad():
			img_high_res = generator_cpu.forward(params)
		img = img_high_res.detach().cpu().numpy()[0]
	
		img = np.clip(img, 0.0, 1.0)
		if save or show:
			# Swap Red with Blue
			img = img[...,[2, 1, 0]]  
			img = np.clip(img, 0.0, 1.0) * 255
		if save:
			image_filename = f"{self._output_dir}/{self._file_basename}.png"
			cv2.imwrite(image_filename, img)
		if show:
			cv2_imshow(img)
			cv2.waitKey()
		return img

	def finish(self):
		"""Finish video writing and save all other data."""
		if self._losses_history:
			losses_filename = f"{self._output_dir}/{self._file_basename}_losses"
			plot_and_save_losses(self._losses_history, 
													 title=f"{self._file_basename} Losses",
													 filename=losses_filename)
		if self._video_steps:
			self._video_writer.close()
		if self._population_video:
			self._population_video_writer.close()
		metadata_filename = f"{self._output_dir}/{self._file_basename}_metadata.py"
		# export_metadata(metadata_filename) #go back! #?

	def _add_video_frames(self, img_batch, losses):
		"""Add images from numpy image batch to video writers.
		
		Args:
			img_batch: numpy array, batch of images (S,H,W,C)
			losses: numpy array, losses for each generator (S,N)
		"""
		if self._video_steps and self._step % self._video_steps == 0:
			# Write image to video.
			best_img = img_batch[np.argmin(losses)]
			self._video_writer.add(cv2.resize(
				best_img, (best_img.shape[1] * 3, best_img.shape[0] * 3)))
			if self._population_video:
				laid_out = layout_img_batch(img_batch)
				self._population_video_writer.add(cv2.resize(
					laid_out, (laid_out.shape[1] * 2, laid_out.shape[0] * 2)))

# ------

#@title CollageTiler class

class CollageTiler():
	def __init__(self,
							 device, dir_results, clip_model, #device is #new #dir_results is #new # clip_model is #new
							 wide, high,
							 prompts,
							 segmented_data, 
							 segmented_data_high_res, 
							 fixed_background_image,
							 background_use,
							 compositional,
							 output_dir,
							 file_basename,
							 video_steps=0,
							 settings={}):
		"""Creates a large collage by producing multiple interlaced collages.

		Args:
			width: number of tiles wide
			height: number of tiles high
			prompts: list of prompts for the collage maker
			segmented_data: patch data for collage maker to use during opmtimisation
			segmented_data_high_res: high res patch data for final renders
			fixed_background_image: highest res background image
			background_use: how to use the background, e.g. per tile or whole image
			compositional: bool, use compositional for multi-CLIP collage tiles
			output_dir: directory for generated files
			file_basename: base name for files
			video_steps: How often to capture frames for videos. Zero=never
		"""
		self._device = device
		self._dir_results = dir_results
		self._clip_model = clip_model
		
		if 'RENDER_METHOD' in settings and settings['RENDER_METHOD'] == 'masked_transparency':
			collage_settings['INITIAL_MIN_RGB'], collage_settings['INITIAL_MAX_RGB'] = 0.7, 1.0 # adjust to new min and max rgb 

		default_settings = collage_settings.copy()
		
		if len(settings):
			for setting, val in settings.items():
				if setting in default_settings: default_settings[setting] = val

		self._settings = default_settings #new #*
		
		self._high_res_multiplier = self._settings['HIGH_RES_MULTIPLIER']
		self._CANVAS_WIDTH = self._settings['CANVAS_WIDTH']
		self._CANVAS_HEIGHT = self._settings['CANVAS_HEIGHT']

		self._tiles_wide = wide
		self._tiles_high = high
		self._prompts = prompts
		self._segmented_data = segmented_data
		self._segmented_data_high_res = segmented_data_high_res
		self._fixed_background_image = fixed_background_image
		self._background_use = background_use
		self._compositional_image = compositional
		self._output_dir = output_dir
		self._file_basename = file_basename
		self._video_steps = video_steps
		
		self._tile_base = "img_tile_y{}_x{}.npy"
		
		# Size of bigger image
		self._tile_width = self._CANVAS_WIDTH
		self._tile_height = self._CANVAS_HEIGHT
		
		self._overlap = 1. / 3.
		
		self._width = int(((2 * self._tiles_wide + 1) * self._tile_width) / 3.)
		self._height = int(((2 * self._tiles_high + 1) * self._tile_height) / 3.)
		
		'''
 		self._width = self._CANVAS_WIDTH #TODO: check if this should instead correspond directly with tile size
 		self._height = self._CANVAS_HEIGHT
 		self._tile_width = int(np.floor(self._width / self._tiles_wide))
 		self._tile_height = int(np.floor(self._height / self._tiles_wide))
		

		if self._compositional_image:
			self._tile_width = 448
			self._tile_height = 448
		else:
			self._high_res_multiplier = 4
			self._tile_width = 224
			self._tile_height = 224

		# Size of bigger image
		self._width = self._tile_width * self._tiles_wide
		self._height = self._tile_height * self._tiles_high
		'''

		self._high_res_tile_width = self._tile_width * self._high_res_multiplier
		self._high_res_tile_height = self._tile_height * self._high_res_multiplier
		self._high_res_width = self._high_res_tile_width * self._tiles_wide
		self._high_res_height = self._high_res_tile_height * self._tiles_high

		self._print_info()
		self._x = 0
		self._y = 0
		self._collage_maker = None
		self._fixed_background = self._scale_fixed_background(high_res=True)

	def _print_info(self):
		print(f"Tiling {self._tiles_wide}x{self._tiles_high} collages")
		print("Optimisation:")
		print(f"Tile size: {self._tile_width}x{self._tile_height}")
		print(f"Global size: {self._width}x{self._height} (WxH)")
		print("High res:")
		print(
			f"Tile size: {self._high_res_tile_width}x{self._high_res_tile_height}")
		print(f"Global size: {self._high_res_width}x{self._high_res_height} (WxH)")
		print(self._prompts)
		for i, tile_prompts in enumerate(self._prompts):
			print(f"Tile {i} prompts: {tile_prompts}")

	def loop(self):
		while self._y < self._tiles_high:
			print(f'\ny tile {self._y} / {self._tiles_high}.') #?
			while self._x < self._tiles_wide:
				print(f'x tile {self._x} / {self._tiles_wide}.') #?
				if not self._collage_maker:
					# Create new collage maker with its unique background.
					print(f"New collage creator for y{self._y}, x{self._x} with bg:")
					tile_bg, self._tile_high_res_bg = self._get_tile_background()
					tile_name = f"{self._file_basename}_y{self._y}_x{self._x}"
					show_and_save(tile_bg, self._dir_results, img_format="SCHW", stitch=False)
					prompts = self._prompts[self._y * self._tiles_wide + self._x]
					self._collage_maker = CollageMaker(
						self._device, self._clip_model, self._dir_results, #new
						prompts, self._segmented_data, 
						tile_bg, self._compositional_image, self._output_dir,
						tile_name, self._video_steps, population_video=False,
						settings=self._settings)
				self._collage_maker.loop()
				collage_img = self._collage_maker.high_res_render(
					self._segmented_data_high_res, 
					self._tile_high_res_bg,
					gamma=1.0, 
					show=True,
					save=True)
				self._save_tile(collage_img / 255)
				# TODO: Currently calling finish will save video and download zip which is not needed.
				# self._collage_maker.finish()
				del self._collage_maker
				self._collage_maker = None
				self._x += 1
			self._y += 1
			self._x = 0
		return collage_img  # SHWC

	def _save_tile(self, img):
		filename = f"final_image_part{str(self._y)}_{str(self._x)}.npy"
		np.save(filename, img)
		background_image_np = np.asarray(img)
		background_image_np = background_image_np[..., ::-1].copy()
		np.save(self._tile_base.format(self._y, self._x), background_image_np)

	def _scale_fixed_background(self, high_res=True):
		if self._fixed_background_image is None:
			return None
		multiplier = self._high_res_multiplier if high_res else 1
		print('_scale_fixed_background... multiplier/high_res/_background_use/self._height', multiplier, high_res, self._background_use, self._height) #remove #debug
		if self._background_use == "Local":
			height = self._tile_height * multiplier
			width = self._tile_width * multiplier
		elif self._background_use == "Global":
			height = self._height * multiplier
			width = self._width * multiplier
		return resize(self._fixed_background_image.astype(float), (height, width))

	def _get_tile_background(self):
		"""Get the background for a particular tile.
	
		This involves getting bordering imagery from left, top left, above and top 
		right, where appropriate.
		i.e. tile (1,1) shares overlap with (0,1), (0,2) and (1,0)
		(0,0), (0,1), (0,2), (0,3)
		(1,0), (1,1), (1,2), (1,3)
		(2,0), (2,1), (2,2), (2,3)
	
		Note that (0,0) is not needed as its contribution is already in (0,1) 
		"""
		if self._fixed_background is None:
			tile_border_bg = np.zeros((self._high_res_tile_height,
																self._high_res_tile_width, 3))
		else:
			if self._background_use == "Local":
				tile_border_bg = self._fixed_background.copy()
			else:  # Crop out section for this tile.
				#orgin_y = self._y * self._high_res_tile_height - int(
				#    self._high_res_tile_height * 2 * self._overlap)
				orgin_y = self._y * (self._high_res_tile_height
														 - int(self._high_res_tile_height * self._overlap))
				orgin_x = self._x * (self._high_res_tile_width 
														 - int(self._high_res_tile_width * self._overlap))
				#orgin_x = self._x * self._high_res_tile_width - int(
				#    self._high_res_tile_width * 2 * self._overlap)
				tile_border_bg = self._fixed_background[
						orgin_y : orgin_y + self._high_res_tile_height,
						orgin_x : orgin_x + self._high_res_tile_width, :]
		tile_idx = dict()
		if self._x > 0:
			tile_idx["left"] = (self._y, self._x - 1)
		if self._y > 0:
			tile_idx["above"] = (self._y - 1, self._x)
			if self._x < self._tiles_wide - 1:  # Penultimate on the row
				tile_idx["above_right"] = (self._y - 1, self._x + 1)

		# Get and insert bodering tile content in this order.
		if "above" in tile_idx:
			self._copy_overlap(tile_border_bg, "above", tile_idx["above"])
		if "above_right" in tile_idx:
			self._copy_overlap(tile_border_bg, "above_right", tile_idx["above_right"])
		if "left" in tile_idx:
			self._copy_overlap(tile_border_bg, "left", tile_idx["left"])

		background_image = self._resize_image_for_torch(
			tile_border_bg, self._tile_height, self._tile_width)
		background_image_high_res = self._resize_image_for_torch(
			tile_border_bg,
			self._high_res_tile_height,
			self._high_res_tile_width).to('cpu')
	
		return background_image, background_image_high_res

	def _resize_image_for_torch(self, img, height, width):
		# Resize and permute to format used by Collage class (SCHW).
		img = torch.tensor(resize(img.astype(float),
															(height, width))).cuda()
		return img.permute(2, 0, 1).to(torch.float32)

	def _copy_overlap(self, target, location, tile_idx):
		# print(
		#     f"Copying overlap from {location} ({tile_idx}) for {self._y},{self._x}")
		big_height = self._high_res_tile_height
		big_width = self._high_res_tile_width
		pixel_overlap = int(big_width * self._overlap)

		# print(f"Loading tile {self._tile_base.format(tile_idx[0], tile_idx[1])}")
		source = np.load(self._tile_base.format(tile_idx[0], tile_idx[1]))
		if location == "above":
			target[0 : pixel_overlap, 0 : big_width, :] = source[
				big_height - pixel_overlap : big_height, 0 : big_width, :]
		if location == "left":
			target[:, 0 : pixel_overlap, :] = source[
				:, big_width - pixel_overlap : big_width, :]
		elif location == "above_right":
			target[0 : pixel_overlap, big_width - pixel_overlap : big_width, :] = source[
				big_height - pixel_overlap : big_height, 0 : pixel_overlap, :]

	def assemble_tiles(self):
		# Stitch together the whole image.
		big_height = self._high_res_tile_height
		big_width = self._high_res_tile_width
		full_height = int((big_height + 2 * big_height * self._tiles_high) / 3)
		full_width = int((big_width + 2 * big_width * self._tiles_wide) / 3)
		full_image = np.zeros((full_height, full_width, 3)).astype('float32')
		
		for y in range(self._tiles_high):
			for x in range(self._tiles_wide):
				tile = np.load(self._tile_base.format(y, x))
				y_offset = int(big_height * y * 2 / 3)
				x_offset = int(big_width * x * 2 / 3)
				full_image[y_offset : y_offset + big_height,
									x_offset : x_offset + big_width, :] = tile[:, :, :]
		show_and_save(
			full_image, self._dir_results, img_format="SHWC", stitch=False, filename="full_image.png")
