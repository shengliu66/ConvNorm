import comet_ml
import argparse
import collections
import sys
import requests
import socket
import torch
import data_loader.data_loaders as module_data
import models.loss as module_loss
import models.metric as module_metric
import models.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from collections import OrderedDict
import random
from utils import set_seed
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn


def log_params(conf: OrderedDict, parent_key: str = None):
    for key, value in conf.items():
        if parent_key is not None:
            combined_key = f'{parent_key}-{key}'
        else:
            combined_key = key

        if not isinstance(value, OrderedDict):
            mlflow.log_param(combined_key, value)
        else:
            log_params(value, combined_key)


def main(config: ConfigParser):

    logger = config.get_logger('train')

    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size= config['data_loader']['args']['batch_size'],
        shuffle=config['data_loader']['args']['shuffle'],
        validation_split=config['data_loader']['args']['validation_split'],
        num_batches=config['data_loader']['args']['num_batches'],
        training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        pin_memory=config['data_loader']['args']['pin_memory'] 
    )

    valid_data_loader = data_loader.split_validation()

    test_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=128,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    ).split_validation()

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)

    train_loss = getattr(module_loss, config['train_loss'])
    val_loss = getattr(module_loss, config['val_loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    logger.info(str(model).split('\n')[-1])

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler

    trainable_params = [{'params': [p for p in model.parameters() if (not getattr(p, 'bin_gate', False)) and (not getattr(p, 'bin_theta', False)) and (not getattr(p, 'srelu_bias', False)) and getattr(p, 'requires_grad', False)]},
              {'params': [p for p in model.parameters() if getattr(p, 'bin_gate', False) and getattr(p, 'requires_grad', False)], 
               'lr': config['optimizer']['args']['lr']*10, 'weight_decay': 0},
               {'params': [p for p in model.parameters() if getattr(p, 'srelu_bias', False) and getattr(p, 'requires_grad', False)], 
                'weight_decay': 0},
                {'params': [p for p in model.parameters() if getattr(p, 'bin_theta', False) and getattr(p, 'requires_grad', False)], 
                'lr': config['optimizer']['args']['lr'], 'weight_decay': 0}
               ]

    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, train_loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler,
                      val_criterion=val_loss)

    trainer.train()
    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    cfg_trainer = config['trainer']


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--percent', '--percent'], type=float, target=('trainer', 'percent')),
        CustomArgs(['--conv', '--conv_layer'], type=str, target=('arch', 'args', 'conv_layer_type')),
        CustomArgs(['--norm', '--norm_layer'], type=str, target=('arch', 'args', 'norm_layer_type')),
        CustomArgs(['--subset_percent', '--subset_percent'], type=float, target=('trainer', 'subset_percent')),
        CustomArgs(['--asym', '--asym'], type=bool, target=('trainer', 'asym')),
        CustomArgs(['--sym', '--sym'], type=bool, target=('trainer', 'sym')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--key', '--comet_key'], type=str, target=('comet','api')),
        CustomArgs(['--offline', '--comet_offline'], type=str, target=('comet','offline')),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',)),
        CustomArgs(['--wd', '--weight_decay'], type=float, target=('optimizer', 'args', 'weight_decay'))
    ]
    config = ConfigParser.get_instance(args, options)

    set_seed(manualSeed = config['seed'])
    main(config)
