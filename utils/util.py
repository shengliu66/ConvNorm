import os
import logging
import random
import numpy as np
import torch
import json
import copy

from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict
from models.prec_conv import Preconditioned_Conv2d

# Below 2 methods from https://github.com/samaonline/Orthogonal-Convolutional-Neural-Networks/blob/14de5261e24544cef78b94c56684d2f1520c1e41/imagenet/utils.py#L34
def deconv_orth_dist(kernel, stride = 2, padding = 1):
    [o_c, i_c, w, h] = kernel.shape
    output = torch.conv2d(kernel, kernel, stride=stride, padding=padding)
    target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).to(kernel.device)
    ct = int(np.floor(output.shape[-1]/2))
    target[:,:,ct,ct] = torch.eye(o_c).to(kernel.device)
    return torch.norm( output - target )
    
def orth_dist(mat, stride=None):
    mat = mat.reshape( (mat.shape[0], -1) )
    if mat.shape[0] < mat.shape[1]:
        mat = mat.permute(1,0)
    return torch.norm( torch.t(mat)@mat - torch.eye(mat.shape[1]).to(mat.device))
                      
def load_from_state_dict(current_model, checkpoint_state):
    """
    Since running_V in preconditioning layers don't have correct init size,
    we manually load them here
    """
    state = copy.deepcopy(checkpoint_state)
    all_precs = []
    all_kernel_v = []
    for key in checkpoint_state.keys():
        if "running_V" in key:
            all_kernel_v.append(checkpoint_state[key])
            state[key] = torch.zeros(1)
    # Load all other parameters
    current_model.load_state_dict(state) #, strict=False
    # Load running_V 's
    itera = 0
    for mod in current_model.modules():
        if isinstance(mod, Preconditioned_Conv2d): #ConvNorm_2d
            mod.running_V = all_kernel_v[itera]
            itera += 1

def get_logger(name, verbosity=2):
    log_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                   log_levels.keys())
    assert verbosity in log_levels, msg_verbosity
    logger = logging.getLogger(name)
    logger.setLevel(log_levels[verbosity])
    return logger

class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()
        
def path_formatter(args):
    # Automatically create a path name for storing checkpoints
    
    args_dict = vars(args)
    check_path = []
    needed_args = ['dataset', 'model', 'lr', 'optimizer', 'batch_size',
                   'lr_scheduler', 'weight_decay', 'norm_method', 'deconv',
                   'seed'] 
    
    for setting in needed_args:
        value = args_dict[setting]
        if value == '':
            value = 'default_scheduler'
        if setting == 'deconv':
            value = 'deconv ' + str(value)
        check_path.append('{}'.format(value))

    timestamp = datetime.datetime.now().strftime("%m-%d-%H.%M")
    check_path.append(timestamp)
    save_path = ','.join(check_path)
    return os.path.join(args.checkpoint_path,save_path).replace("\\","/")

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def set_seed(manualSeed=666):
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(manualSeed)

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from  2"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    current = np.clip(current, 0.0, rampdown_length)
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def cosine_rampup(current, rampup_length):
    """Cosine rampup"""
    current = np.clip(current, 0.0, rampup_length)
    return float(-.5 * (np.cos(np.pi * current / rampup_length) - 1))


