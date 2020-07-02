import torch.nn as nn
import torch.nn.functional as F

class InterpolateModule(nn.Module):
	"""
	This is a module version of F.interpolate (rip nn.Upsampling).
	Any arguments you give it just get passed along for the ride.
	"""

	def __init__(self, *args, **kwdargs):
            super().__init__()

            self.size = kwdargs.get('size', None)
            self.scale_factor = kwdargs.get('scale_factor', None)
            self.mode = kwdargs.get('mode', 'nearest')
            self.align_corners = kwdargs.get('align_corners', None)
            self.recompute_scale_factor = kwdargs.get('recompute_scale_factor', None)

	def forward(self, x):
		return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners, recompute_scale_factor=self.recompute_scale_factor)
