"""
This file defines the core research contribution
"""
import matplotlib
matplotlib.use('Agg')
import math

import torch
from torch import nn
from models.psp.encoders import psp_encoders
from models.psp.stylegan2.model import Generator
from models.configs.paths_config import model_paths


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class pSp(nn.Module):

	def __init__(self):
		super(pSp, self).__init__()
		self.encoder_type = "GradualStyleEncoder"

		# compute number of style inputs based on the output resolution
		self.n_styles = int(math.log(256, 2)) * 2 - 2
		# Define architecture
		self.encoder = self.set_encoder()
		self.decoder = Generator(256, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		self.latent_avg = None
		# Load weights if needed
		# self.load_weights()

	def set_encoder(self):
		if self.encoder_type == 'GradualStyleEncoder':
			encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', 1)
		else:
			raise Exception('{} is not a valid encoders')
		return encoder

	

	def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			# normalize with respect to the center of an average face
			# if self.opts.start_from_latent_avg:
			# 	if self.opts.learn_in_w:
			# 		codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
			# 	else:
			# 		codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)


		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code
		images, result_latent = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)

		if resize:
			images = self.face_pool(images)

		if return_latents:
			return images, result_latent
		else:
			return images


	
