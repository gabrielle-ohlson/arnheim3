#@title Imports:
import numpy as np
import torch
import torch.nn.functional as F
from kornia.color import hsv

# ------

#@title Affine transform settings

affine_transform_settings = {
	# Translation bounds for X and Y.
	'MIN_TRANS': -0.66, # (min:-1.0, max:1.0)
	'MAX_TRANS': 0.8, # (min:-1.0, max:1.0)
	# Scale bounds (> 1 means zoom out and < 1 means zoom in).
	'MIN_SCALE': 1,
	'MAX_SCALE': 2,
	# Bounds on ratio between X and Y scale (default 1).
	'MIN_SQUEEZE': 0.5,
	'MAX_SQUEEZE': 2.0,
	# Shear deformation bounds (default 0)
	'MIN_SHEAR': -0.2, # (min:-1.0, max:1.0)
	'MAX_SHEAR': 0.2, # (min:-1.0, max:1.0)
	# Rotation bounds.
	'MIN_ROT_DEG': -180, # (min:-180, max:180)
	'MAX_ROT_DEG': 180, # (min:-180, max:180)

	### Mutation levels
	# Scale mutation applied to position and rotation, scale, distortion, colour and patch swaps.
	'POS_AND_ROT_MUTATION_SCALE': 0.02, # (min:0.0, max:0.3) #affine
	'SCALE_MUTATION_SCALE': 0.02, # (min:0.0, max:0.3) #affine
	'DISTORT_MUTATION_SCALE': 0.02, # (min:0.0, max:0.3)
	'COLOUR_MUTATION_SCALE': 0.02 # (min:0.0, max:0.3)
}

affine_transform_settings['MIN_ROT'] = affine_transform_settings['MIN_ROT_DEG'] * np.pi / 180.0
affine_transform_settings['MAX_ROT'] = affine_transform_settings['MAX_ROT_DEG'] * np.pi / 180.0 

# ------
#@title Evolution settings

# Reasonable defaults:
# POP_SIZE = 2
# EVOLUTION_FREQUENCY = 100
# MUTATION_SCALES = ~0.1
# MAX_MULTIPLE_VISUALISATIONS = 7

# mutation_settings = {
#     ### Mutation levels
#     # Scale mutation applied to position and rotation, scale, distortion, colour and patch swaps.
#     'POS_AND_ROT_MUTATION_SCALE': 0.02, # (min:0.0, max:0.3) #affine
#     'SCALE_MUTATION_SCALE': 0.02, # (min:0.0, max:0.3) #affine
#     'DISTORT_MUTATION_SCALE': 0.02, # (min:0.0, max:0.3)
#     'COLOUR_MUTATION_SCALE': 0.02 # (min:0.0, max:0.3)
# }


# ------

#@title Colour transform settings

colour_transform_settings = {
	# RGB
	'MIN_RGB': -0.21, # (min: -1, max: 1)
	'MAX_RGB': 1.0, # (min: 0, max: 1)
	'INITIAL_MIN_RGB': 0.05, # (min: 0, max: 1)
	'INITIAL_MAX_RGB': 0.25, # (min: 0, max: 1)
	# HSV
	'MIN_HUE': 0., # (min: 0, max: 1)
	'MAX_HUE_DEG': 360, # (min: 0, max: 360)
	# MAX_HUE = MAX_HUE_DEG * np.pi / 180.0
	'MIN_SAT': 0., # (min: 0, max: 1)
	'MAX_SAT': 1.,  # (min: 0, max: 1)
	'MIN_VAL': 0., # (min: -1, max: 1)
	'MAX_VAL': 1. # (min: 0, max: 1)
}

colour_transform_settings['MAX_HUE'] = colour_transform_settings['MAX_HUE_DEG'] * np.pi / 180.0

# ------

#@title Affine transform classes

class PopulationAffineTransforms(torch.nn.Module):
	"""Population-based Affine Transform network."""
	def __init__(self, device, num_patches=1, pop_size=1, settings={}): #device is #new
		super(PopulationAffineTransforms, self).__init__()

		self._device = device

		self._num_patches = num_patches #new

		default_settings = affine_transform_settings.copy()

		print('settings:', settings, 'default_settings': default_settings) #remove #debug

		if len(settings):
			for setting, val in settings.items():
				if setting in default_settings: default_settings.setting = val

		self._settings = default_settings #new #*

		print('self._settings:', self._settings) #remove #debug

		self._pop_size = pop_size
		matrices_translation = (
			np.random.rand(pop_size, num_patches, 2, 1) * (self._settings['MAX_TRANS'] - self._settings['MIN_TRANS']) 
			+ self._settings['MIN_TRANS'])
		matrices_rotation = (
			np.random.rand(pop_size, num_patches, 1, 1) * (self._settings['MAX_ROT'] - self._settings['MIN_ROT'])
			+ self._settings['MIN_ROT'])
		matrices_scale = (
			np.random.rand(pop_size, num_patches, 1, 1) * (self._settings['MAX_SCALE'] - self._settings['MIN_SCALE']) 
			+ self._settings['MIN_SCALE'])
		matrices_squeeze = (
			np.random.rand(pop_size, num_patches, 1, 1) * (
				(self._settings['MAX_SQUEEZE'] - self._settings['MIN_SQUEEZE']) + self._settings['MIN_SQUEEZE']))
		matrices_shear = (
			np.random.rand(pop_size, num_patches, 1, 1) * (self._settings['MAX_SHEAR'] - self._settings['MIN_SHEAR']) 
			+ self._settings['MIN_SHEAR'])
		self.translation = torch.nn.Parameter(
			torch.tensor(matrices_translation, dtype=torch.float),
			requires_grad=True)
		self.rotation = torch.nn.Parameter(
			torch.tensor(matrices_rotation, dtype=torch.float),
			requires_grad=True)
		self.scale = torch.nn.Parameter(
			torch.tensor(matrices_scale, dtype=torch.float),
			requires_grad=True)
		self.squeeze = torch.nn.Parameter(
			torch.tensor(matrices_squeeze, dtype=torch.float),
			requires_grad=True)
		self.shear = torch.nn.Parameter(
			torch.tensor(matrices_shear, dtype=torch.float),
			requires_grad=True)
		self._identity = (
			torch.ones((pop_size, num_patches, 1, 1)) * torch.eye(2).unsqueeze(0)
			).to(device)
		self._zero_column = torch.zeros((pop_size, num_patches, 2, 1)).to(device)
		self._unit_row = (
			torch.ones((pop_size, num_patches, 1, 1)) * torch.tensor([0., 0., 1.])
			).to(device)
		self._zeros = torch.zeros((pop_size, num_patches, 1, 1)).to(device)

	def _clamp(self):
		self.translation.data = self.translation.data.clamp(
			min=self._settings['MIN_TRANS'], max=self._settings['MAX_TRANS'])
		self.rotation.data = self.rotation.data.clamp(
			min=self._settings['MIN_ROT'], max=self._settings['MAX_ROT'])
		self.scale.data = self.scale.data.clamp(
			min=self._settings['MIN_SCALE'], max=self._settings['MAX_SCALE'])
		self.squeeze.data = self.squeeze.data.clamp(
			min=self._settings['MIN_SQUEEZE'], max=self._settings['MAX_SQUEEZE'])
		self.shear.data = self.shear.data.clamp(
			min=self._settings['MIN_SHEAR'], max=self._settings['MAX_SHEAR'])

	def copy_and_mutate_s(self, parent, child):
		"""Copy parameters to child, mutating transform parameters."""
		device = self._device

		with torch.no_grad():
			self.translation[child, ...] = (self.translation[parent, ...] 
				+ self._settings['POS_AND_ROT_MUTATION_SCALE'] * torch.randn(
					self.translation[child, ...].shape).to(device))
			self.rotation[child, ...] = (self.rotation[parent, ...]  
				+ self._settings['POS_AND_ROT_MUTATION_SCALE'] * torch.randn(
					self.rotation[child, ...].shape).to(device))
			self.scale[child, ...] = (self.scale[parent, ...] 
				+ self._settings['SCALE_MUTATION_SCALE'] * torch.randn(
					self.scale[child, ...].shape).to(device))
			self.squeeze[child, ...] = (self.squeeze[parent, ...]
				+ self._settings['DISTORT_MUTATION_SCALE'] * torch.randn(
					self.squeeze[child, ...].shape).to(device))
			self.shear[child, ...] = (self.shear[parent, ...]
				+ self._settings['DISTORT_MUTATION_SCALE'] * torch.randn(
					self.shear[child, ...].shape).to(device))

	def copy_from(self, other, idx_to, idx_from):
		"""Copy parameters from other spatial transform, for selected indices."""
		assert idx_to < self._pop_size
		with torch.no_grad():
			self.translation[idx_to, ...] = other.translation[idx_from, ...]
			self.rotation[idx_to, ...] = other.rotation[idx_from, ...]
			self.scale[idx_to, ...] = other.scale[idx_from, ...]
			self.squeeze[idx_to, ...] = other.squeeze[idx_from, ...]
			self.shear[idx_to, ...] = other.shear[idx_from, ...]

	def forward(self, x):
		self._clamp()
		scale_affine_mat = torch.cat([
			torch.cat([self.scale, self.shear], 3),
			torch.cat([self._zeros, self.scale * self.squeeze], 3)],
			2)
		scale_affine_mat = torch.cat([
			torch.cat([scale_affine_mat, self._zero_column], 3),
			self._unit_row], 2)
		rotation_affine_mat = torch.cat([
			torch.cat([torch.cos(self.rotation), -torch.sin(self.rotation)], 3),
			torch.cat([torch.sin(self.rotation), torch.cos(self.rotation)], 3)],
			2)
		rotation_affine_mat = torch.cat([
			torch.cat([rotation_affine_mat, self._zero_column], 3),
			self._unit_row], 2)
		
		scale_rotation_mat = torch.matmul(scale_affine_mat,
																			rotation_affine_mat)[:, :, :2, :]
		# Population and patch dimensions (0 and 1) need to be merged.
		# E.g. from (POP_SIZE, NUM_PATCHES, CHANNELS, WIDTH, HEIGHT) 
		# to (POP_SIZE * NUM_PATCHES, CHANNELS, WIDTH, HEIGHT)
		scale_rotation_mat = scale_rotation_mat[:, :, :2, :].view(
			1, -1, *(scale_rotation_mat[:, :, :2, :].size()[2:])).squeeze()
		x = x.view(1, -1, *(x.size()[2:])).squeeze()
		scaled_rotated_grid = F.affine_grid(
			scale_rotation_mat, x.size(), align_corners=True)
		scaled_rotated_x = F.grid_sample(x, scaled_rotated_grid, align_corners=True)

		translation_affine_mat = torch.cat([self._identity, self.translation], 3)
		translation_affine_mat = translation_affine_mat.view(
			1, -1, *(translation_affine_mat.size()[2:])).squeeze()
		translated_grid = F.affine_grid(
			translation_affine_mat, x.size(), align_corners=True)
		y = F.grid_sample(scaled_rotated_x, translated_grid, align_corners=True)
		# return y.view(self._pop_size, NUM_PATCHES, *(y.size()[1:]))
		return y.view(self._pop_size, self._num_patches, *(y.size()[1:]))

	def tensor_to(self, device):
		self.translation = self.translation.to(device)
		self.rotation = self.rotation.to(device)
		self.scale = self.scale.to(device)
		self.squeeze = self.squeeze.to(device)
		self.shear = self.shear.to(device)
		self._identity = self._identity.to(device)
		self._zero_column = self._zero_column.to(device)
		self._unit_row = self._unit_row.to(device)
		self._zeros = self._zeros.to(device)

# ------
#@title RGB and HSV color transforms

class PopulationOrderOnlyTransforms(torch.nn.Module):

	def __init__(self, device, num_patches=1, pop_size=1):
		super(PopulationOrderOnlyTransforms, self).__init__()

		self._pop_size = pop_size

		population_zeros = np.ones((pop_size, num_patches, 1, 1, 1))
		population_orders = np.random.rand(pop_size, num_patches, 1, 1, 1)

		self._zeros = torch.nn.Parameter(
			torch.tensor(population_zeros, dtype=torch.float),
			requires_grad=False)
		self.orders = torch.nn.Parameter(
			torch.tensor(population_orders, dtype=torch.float),
			requires_grad=True)
		self._hsv_to_rgb = hsv.HsvToRgb()

	def _clamp(self):
		self.orders.data = self.orders.data.clamp(min=0.0, max=1.0)

	def copy_and_mutate_s(self, parent, child):
		with torch.no_grad():
			self.orders[child, ...] = self.orders[parent, ...]

	def copy_from(self, other, idx_to, idx_from):
		"""Copy parameters from other colour transform, for selected indices."""
		assert idx_to < self._pop_size
		with torch.no_grad():
			self.orders[idx_to, ...] = other.orders[idx_from, ...]

	def forward(self, x):
		self._clamp()
		colours = torch.cat(
			[self._zeros, self._zeros, self._zeros, self._zeros, self.orders],
			2)
		return colours * x

	def tensor_to(self, device):
		self.orders = self.orders.to(device)
		self._zeros = self._zeros.to(device)


class PopulationColourHSVTransforms(torch.nn.Module):

	def __init__(self, device, num_patches=1, pop_size=1, settings={}): #device is #new
		super(PopulationColourHSVTransforms, self).__init__()

		self._device = device

		default_settings = colour_transform_settings.copy()
		if len(settings):
			for setting, val in settings.items():
				if setting in default_settings: default_settings.setting = val
		
		self._settings = default_settings

		print('PopulationColourHSVTransforms for {} patches, {} individuals'.format(
			num_patches, pop_size))
		self._pop_size = pop_size

		coeff_hue = 0.5 * (self._settings['MAX_HUE'] - self._settings['MIN_HUE']) + self._settings['MIN_HUE']
		coeff_sat = 0.5 * (self._settings['MAX_SAT'] - self._settings['MIN_SAT']) + self._settings['MIN_SAT']
		coeff_val = 0.5 * (self._settings['MAX_VAL'] - self._settings['MIN_VAL']) + self._settings['MIN_VAL']
		population_hues = np.random.rand(pop_size, num_patches, 1, 1, 1) * coeff_hue
		population_saturations = np.random.rand(
			pop_size, num_patches, 1, 1, 1) * coeff_sat
		population_values = np.random.rand(
			pop_size, num_patches, 1, 1, 1) * coeff_val
		population_zeros = np.ones((pop_size, num_patches, 1, 1, 1))
		population_orders = np.random.rand(pop_size, num_patches, 1, 1, 1)
		
		self.hues = torch.nn.Parameter(
			torch.tensor(population_hues, dtype=torch.float),
			requires_grad=True)
		self.saturations = torch.nn.Parameter(
			torch.tensor(population_saturations, dtype=torch.float),
			requires_grad=True)
		self.values = torch.nn.Parameter(
			torch.tensor(population_values, dtype=torch.float),
			requires_grad=True)
		self._zeros = torch.nn.Parameter(
			torch.tensor(population_zeros, dtype=torch.float),
			requires_grad=False)
		self.orders = torch.nn.Parameter(
			torch.tensor(population_orders, dtype=torch.float),
			requires_grad=True)
		self._hsv_to_rgb = hsv.HsvToRgb()

	def _clamp(self):
		self.hues.data = self.hues.data.clamp(min=self._settings['MIN_HUE'], max=self._settings['MAX_HUE'])
		self.saturations.data = self.saturations.data.clamp(
			min=self._settings['MIN_SAT'], max=self._settings['MAX_SAT'])
		self.values.data = self.values.data.clamp(min=self._settings['MIN_VAL'], max=self._settings['MAX_VAL'])
		self.orders.data = self.orders.data.clamp(min=0.0, max=1.0)

	def copy_and_mutate_s(self, parent, child):
		device = self._device

		with torch.no_grad():
			self.hues[child, ...] = (
				self.hues[parent, ...]
				+ self._settings['COLOUR_MUTATION_SCALE'] * torch.randn(
					self.hues[child, ...].shape).to(device))
			self.saturations[child, ...] = (
				self.saturations[parent, ...]
				+ self._settings['COLOUR_MUTATION_SCALE'] * torch.randn(
					self.saturations[child, ...].shape).to(device))
			self.values[child, ...] = (
				self.values[parent, ...]
				+ self._settings['COLOUR_MUTATION_SCALE'] * torch.randn(
					self.values[child, ...].shape).to(device))
			self.orders[child, ...] = self.orders[parent, ...]

	def copy_from(self, other, idx_to, idx_from):
		"""Copy parameters from other colour transform, for selected indices."""
		assert idx_to < self._pop_size
		with torch.no_grad():
			self.hues[idx_to, ...] = other.hues[idx_from, ...]
			self.saturations[idx_to, ...] = other.saturations[idx_from, ...]
			self.values[idx_to, ...] = other.values[idx_from, ...]
			self.orders[idx_to, ...] = other.orders[idx_from, ...]

	def forward(self, image):
		self._clamp()
		colours = torch.cat(
			[self.hues, self.saturations, self.values, self._zeros, self.orders], 2)
		hsv_image = colours * image
		rgb_image = self._hsv_to_rgb(hsv_image[:, :, :3, :, :])
		return torch.cat([rgb_image, hsv_image[:, :, 3:, :, :]], axis=2)

	def tensor_to(self, device):
		self.hues = self.hues.to(device)
		self.saturations = self.saturations.to(device)
		self.values = self.values.to(device)
		self.orders = self.orders.to(device)
		self._zeros = self._zeros.to(device)


class PopulationColourRGBTransforms(torch.nn.Module):

	def __init__(self, device, num_patches=1, pop_size=1, settings={}): #device is #new
		super(PopulationColourRGBTransforms, self).__init__()

		self._device = device

		default_settings = colour_transform_settings.copy()

		if len(settings):
			for setting, val in settings.items():
				if setting in default_settings: default_settings.setting = val
		
		self._settings = default_settings

		print('PopulationColourRGBTransforms for {} patches, {} individuals'.format(
				num_patches, pop_size))
		self._pop_size = pop_size

		rgb_init_range = self._settings['INITIAL_MAX_RGB'] - self._settings['INITIAL_MIN_RGB']
		population_reds = (np.random.rand(pop_size, num_patches, 1, 1, 1) 
			* rgb_init_range) + self._settings['INITIAL_MIN_RGB']
		population_greens = (np.random.rand(
			pop_size, num_patches, 1, 1, 1) * rgb_init_range) + self._settings['INITIAL_MIN_RGB']
		population_blues = (np.random.rand(
			pop_size, num_patches, 1, 1, 1) * rgb_init_range) + self._settings['INITIAL_MIN_RGB']
		population_zeros = np.ones((pop_size, num_patches, 1, 1, 1))
		population_orders = np.random.rand(pop_size, num_patches, 1, 1, 1)

		self.reds = torch.nn.Parameter(
			torch.tensor(population_reds, dtype=torch.float),
			requires_grad=True)
		self.greens = torch.nn.Parameter(
			torch.tensor(population_greens, dtype=torch.float),
			requires_grad=True)
		self.blues = torch.nn.Parameter(
			torch.tensor(population_blues, dtype=torch.float),
			requires_grad=True)
		self._zeros = torch.nn.Parameter(
			torch.tensor(population_zeros, dtype=torch.float),
			requires_grad=False)
		self.orders = torch.nn.Parameter(
			torch.tensor(population_orders, dtype=torch.float),
			requires_grad=True)

	def _clamp(self):
		self.reds.data = self.reds.data.clamp(min=self._settings['MIN_RGB'], max=self._settings['MAX_RGB'])
		self.greens.data = self.greens.data.clamp(min=self._settings['MIN_RGB'], max=self._settings['MAX_RGB'])
		self.blues.data = self.blues.data.clamp(min=self._settings['MIN_RGB'], max=self._settings['MAX_RGB'])
		self.orders.data = self.orders.data.clamp(min=0.0, max=1.0)

	def copy_and_mutate_s(self, device, parent, child): #device is #new
		device = self._device

		with torch.no_grad():
			self.reds[child, ...] = (
				self.reds[parent, ...] 
				+ self._settings['COLOUR_MUTATION_SCALE'] * torch.randn(
					self.reds[child, ...].shape).to(device))
			self.greens[child, ...] = (
				self.greens[parent, ...] 
				+ self._settings['COLOUR_MUTATION_SCALE'] * torch.randn(
					self.greens[child, ...].shape).to(device))
			self.blues[child, ...] = (
				self.blues[parent, ...] 
				+ self._settings['COLOUR_MUTATION_SCALE'] * torch.randn(
					self.blues[child, ...].shape).to(device))
			self.orders[child, ...] = self.orders[parent, ...] 

	def copy_from(self, other, idx_to, idx_from):
		"""Copy parameters from other colour transform, for selected indices."""
		assert idx_to < self._pop_size
		with torch.no_grad():
			self.reds[idx_to, ...] = other.reds[idx_from, ...]
			self.greens[idx_to, ...] = other.greens[idx_from, ...]
			self.blues[idx_to, ...] = other.blues[idx_from, ...]
			self.orders[idx_to, ...] = other.orders[idx_from, ...]

	def forward(self, x):
		self._clamp()
		colours = torch.cat(
			[self.reds, self.greens, self.blues, self._zeros, self.orders], 2)
		return colours * x

	def tensor_to(self, device):
		self.reds = self.reds.to(device)
		self.greens = self.greens.to(device)
		self.blues = self.blues.to(device)
		self.orders = self.orders.to(device)
		self._zeros = self._zeros.to(device)
