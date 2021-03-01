import torch
import torch.fft
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single, _pair, _triple

def complex_abs(tensor):
    return (tensor**2).sum(-1)

def compl_mul_2D(a, b):
    """
    Given a and b two tensors of dimension 5
    with the last dimension being the real and imaginary part,
    """
    return torch.stack([
        a[..., 0].mul_(b),
        a[..., 1].mul_(b)], dim=-1)

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    current = np.clip(current, 0.0, rampdown_length)
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

class PreConv(_ConvNd):
    """
    Preconditioned convolution
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0,
        dilation=1, groups=1, bias=False, padding_mode='zeros', affine = True,
        bn = True, momentum = None , track_running_stats=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(PreConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.affine = affine
        self.bn = bn
        self.register_buffer('running_V', torch.zeros(1))
        self.track_running_stats = track_running_stats
        self.num_batches_tracked = 0
        self.momentum = momentum
        k = kernel_size[0]
        if self.bn:
            self.batch_norm = nn.BatchNorm2d(out_channels, affine = affine)
        if affine:
            self.bpconv = nn.Conv2d(out_channels, out_channels, k, padding = (k-1) // 2, groups=out_channels, bias=True)
        else:
            self.bpconv = nn.Sequential()
    
    def _truncate_circ_to_cross(self, out):
        # First calculate how much to truncate
        out_sizex_start = self.kernel_size[0] - 1 - self.padding[0]
        out_sizey_start = self.kernel_size[1] - 1 - self.padding[1]
        if out_sizex_start != 0 and out_sizey_start != 0:
            out = out[:, :, out_sizex_start: -out_sizex_start, out_sizey_start:-out_sizey_start]
        elif out_sizex_start == 0:
            if out_sizey_start != 0:
                out = out[:, :, :, out_sizey_start:-out_sizey_start]
        elif out_sizey_start == 0:
            if out_sizex_start != 0:
                out = out[:, :, out_sizex_start: -out_sizex_start, :]
        # Also considering stride
        if self.stride[0] > 1:
            out = out[..., ::self.stride[0], ::self.stride[1]]
        return out
    
    def _calculate_running_estimate(self, current_V):
        with torch.no_grad():
            exponential_average_factor = 0.0
            if self.track_running_stats:
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = cosine_rampdown(self.num_batches_tracked,40000)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum
            self.running_V = exponential_average_factor * current_V\
                            + (1 - exponential_average_factor) * self.running_V
        
    def conv2d_forward(self, input, weight, stride=1):
        # padding should be kernel size - 1
        padd = (self.kernel_size[0] - 1, self.kernel_size[1] - 1)
        return F.conv2d(input, weight, self.bias, stride,
                        padd, self.dilation, self.groups)
    
    def preconditioning(self, cout, kernel):
        if (self.kernel_size[0]==1) or (self.kernel_size[1] ==1):
            V = kernel ** 2
            V = torch.sum(V, dim=1)
            V = V + 1e-4 #* V.view(V.size(0),-1).max(dim=-1)[0][:,None,None]
            V = torch.exp(-0.5*torch.log(V))
            with torch.no_grad():
                if self.training:
                    self._calculate_running_estimate(V)
                else:
                    V = self.running_V
            return V*cout 
        else:
            final_size_x = cout.size(-2)
            final_size_y = cout.size(-1)
            f_input = torch.rfft(cout, 2, normalized=False, onesided=True)
            with torch.no_grad():
                if self.training:
                    pad_kernel = F.pad(kernel, [0, final_size_y-self.kernel_size[1], 0, final_size_x-self.kernel_size[0]])
                    f_weight = torch.rfft(pad_kernel, 2, normalized=False, onesided=True)
                    V = complex_abs(f_weight) 
                    V = torch.sum(V, dim=1)
                    V = V + 1e-4 #* V.view(V.size(0),-1).max(dim=-1)[0][:,None,None]
                    V = torch.exp(-0.5*torch.log(V))
                    self._calculate_running_estimate(V)
                else:
                    V = self.running_V
            output = torch.irfft(compl_mul_2D(f_input, V), 2, normalized=False, signal_sizes=(final_size_x,final_size_y))
            return output
        
    def forward(self, input):
        c_out = self.conv2d_forward(input, self.weight)
        p_out = self.preconditioning(c_out, self.weight.data.detach())
        # Truncate the preconditioning result for desired spatial size and stride
        p_out = self._truncate_circ_to_cross(p_out)
        # Affine part
        output =  self.bpconv(p_out)
        # If use BatchNorm
        if self.bn:
            output = self.batch_norm(output)
        return output

class Preconditioned_Conv2d(PreConv):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))
