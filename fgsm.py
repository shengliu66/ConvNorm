import torch
import sys
from torch.autograd import Variable
import numpy as np
import time
import torch.nn as nn
from tqdm import tqdm
import models.model as module_arch
#from tqdm import tqdm_notebook as tqdm
from typing import List
import sys
from base import BaseTrainer
from utils import inf_loop, get_logger, Timer, load_from_state_dict, set_seed
from collections import OrderedDict
import argparse
from parse_config import ConfigParser
import data_loader.data_loaders as module_data

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        
        
def fgsm(gradz, step_size):
    return step_size*torch.sign(gradz)


def validate_fgsm(test_loader, model, configs):

    eps = configs['trainer']['adv_clip_eps']
    
    # Attack amount
    color_value = 255.0
    eps /= color_value
    print(f"FGSM attack with capacity {eps}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Mean/Std for normalization   
    # Data mean and std, (cifar10)
    dmean = torch.tensor([0.4914, 0.4822, 0.4465]).to(device)
    dstd = torch.tensor([0.2023, 0.1994, 0.2010]).to(device)

    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for i, (input, target) in enumerate(test_loader):
        
        input = input.to(device)
        target = target.to(device)
        
        orig_input = input.clone()

        invar = Variable(input, requires_grad=True)
        in1 = invar - dmean[None,:,None,None]
        in1.div_(dstd[None,:,None,None])
        output = model(in1)
        ascend_loss = criterion(output, target)
        ascend_grad = torch.autograd.grad(ascend_loss, invar)[0]
        pert = fgsm(ascend_grad, eps)
        # Apply purturbation
        input += pert.data
        input = torch.max(orig_input-eps, input)
        input = torch.min(orig_input+eps, input)
        input.clamp_(0, 1.0)
        
        input.sub_(dmean[None,:,None,None]).div_(dstd[None,:,None,None])
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            if i == 0 or (i + 1) % 50 == 0:
                print('FGSM Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(test_loader), loss=losses,
                       top1=top1, top5=top5))
                sys.stdout.flush()

    print(' FGSM Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return top1.avg


def main(args, config: ConfigParser):

    test_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=100,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    ).split_validation()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model architecture, then print to console
    model = getattr(module_arch, config["arch"]["type"])(
        num_classes = config["arch"]["args"]["num_classes"],
        norm_layer_type = config["arch"]["args"]["norm_layer_type"],
        conv_layer_type = config["arch"]["args"]["conv_layer_type"],
        linear_layer_type = config["arch"]["args"]["linear_layer_type"],
        activation_layer_type = config["arch"]["args"]["activation_layer_type"]
    ).to(device)

    checkpoint = torch.load(args.checkpoint_path)
    #model.load_state_dict(checkpoint["state_dict"])
    load_from_state_dict(model, checkpoint["state_dict"])
    model.eval()

    validate_fgsm(test_data_loader, model, config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='config file path (default: None)')
    parser.add_argument('-s', '--checkpoint_path', type=str, required=True,
                        help='path to find model checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--result_dir', type=str, default='saved', 
                        help='directory for saving results')
    parser.add_argument('--seed', type=int, default=6, 
                        help='Random seed')
    parser.add_argument('--name', type=str, default='', 
                        help='name for this model')

    config = ConfigParser.get_instance(parser)
    args = parser.parse_args()
    print("Start")
    set_seed(manualSeed = args.seed)
    main(args, config)
