import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm
from typing import List
import sys
from base import BaseTrainer
from utils import inf_loop, get_logger, Timer, deconv_orth_dist, orth_dist
from collections import OrderedDict
import argparse


class Adv_Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, train_criterion, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, len_epoch=None, val_criterion=None):
        super().__init__(model, metrics, optimizer, config, val_criterion)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader

        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_loss_list: List[float] = []
        self.val_loss_list: List[float] = []
        self.test_loss_list: List[float] = []

        self.train_criterion = train_criterion

        #Visdom visualization
        self.new_best_val = False
        self.val_acc = 0
        self.test_val_acc = 0
        
        # Data mean and std, (cifar10)
        self.dmean = torch.tensor([0.4914, 0.4822, 0.4465]).to(self.device)
        self.dstd = torch.tensor([0.2023, 0.1994, 0.2010]).to(self.device)
        
        cfg_trainer = config['trainer']
        # OCNN related
        self.ocnn = cfg_trainer["OCNN"]
        print(self.ocnn)
        if self.ocnn:
            self.lamb_ocnn = cfg_trainer["lamb_ocnn"]
            print(f"Adding additional regularization for OCNN with lamb {self.lamb_ocnn}")
        
        # Attack amount
        color_value = 255.0
        self.fgsm_step = cfg_trainer["fgsm_step"] / color_value
        self.adv_clip_eps = cfg_trainer["adv_clip_eps"] / color_value
        print(f"Each fgsm_step is {self.fgsm_step}, total eps {self.adv_clip_eps}")
        self.adv_repeats = cfg_trainer["adv_repeats"]

    def _eval_metrics(self, output, label):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, label)
            self.writer.add_scalar({'{}'.format(metric.__name__): acc_metrics[i]})
        return acc_metrics
    
    def fgsm(self, gradz, step_size):
        return step_size*torch.sign(gradz)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        
        with tqdm(self.data_loader) as progress:
            for batch_idx, (data, label) in enumerate(progress):
                progress.set_description_str(f'Train epoch {epoch}')
                
                data, label = data.to(self.device), label.long().to(self.device)
                
                # Adv training steps
                global_noise_data = torch.zeros([data.shape[0], 3, 32, 32]).to(self.device)
                for j in range(self.adv_repeats):
                    # Ascend on the global noise
                    noise_batch = global_noise_data.clone().requires_grad_(True).to(self.device)
                    in1 = data + noise_batch
                    in1.clamp_(0, 1.0)
                    in1.sub_(self.dmean[None,:,None,None]).div_(self.dstd[None,:,None,None])
                    output = self.model(in1)
                    
                    loss = self.train_criterion(output, label)
                    
                    if self.ocnn:
                        #####
                        # from https://github.com/samaonline/Orthogonal-Convolutional-Neural-Networks/blob/master/imagenet/main_orth18.py
                        # And only applicable to ResNet18
                        diff = orth_dist(self.model.layer2[0].shortcut[0].weight) + orth_dist(self.model.layer3[0].shortcut[0].weight) + orth_dist(self.model.layer4[0].shortcut[0].weight)
                        diff += deconv_orth_dist(self.model.layer1[0].conv1.weight, stride=1) + deconv_orth_dist(self.model.layer1[1].conv1.weight, stride=1)
                        diff += deconv_orth_dist(self.model.layer2[0].conv1.weight, stride=2) + deconv_orth_dist(self.model.layer2[1].conv1.weight, stride=1)
                        diff += deconv_orth_dist(self.model.layer3[0].conv1.weight, stride=2) + deconv_orth_dist(self.model.layer3[1].conv1.weight, stride=1)
                        diff += deconv_orth_dist(self.model.layer4[0].conv1.weight, stride=2) + deconv_orth_dist(self.model.layer4[1].conv1.weight, stride=1)
                        #####
                        loss = loss + self.lamb_ocnn * diff
                    
                    self.writer.set_step((epoch - 1) * self.len_epoch * self.adv_repeats + batch_idx * self.adv_repeats + j, epoch=epoch)
                    self.writer.add_scalar({'loss': loss.item()})
                    self.train_loss_list.append(loss.item())
                    total_loss += loss.item()
                    total_metrics += self._eval_metrics(output, label)

                    if batch_idx % self.log_step == 0:
                        progress.set_postfix_str(' {} Loss: {:.6f}'.format(
                            self._progress(batch_idx),
                            loss.item()))

                    # compute gradient and do SGD step
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Update the noise for the next iteration
                    pert = self.fgsm(noise_batch.grad, self.fgsm_step)
                    global_noise_data[0:data.shape[0]] += pert.data
                    global_noise_data.clamp_(-self.adv_clip_eps, self.adv_clip_eps)

                    self.optimizer.step()
                    
                    # Back to normal set-up
                    for p in self.bin_gates:
                        p.data.clamp_(min=0, max=1)


                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(' {} Loss: {:.6f}'.format(
                        self._progress(batch_idx),
                        loss.item()))
                    #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch:
                    break

        log = {
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist(),
            'learning rate': self.lr_scheduler.get_lr()
        }


        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)
        if self.do_test:
            test_log = self._test_epoch(epoch)
            log.update(test_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            with tqdm(self.valid_data_loader) as progress:
                for batch_idx, (data, label) in enumerate(progress):
                    progress.set_description_str(f'Valid epoch {epoch}')
                    data, label = data.to(self.device), label.to(self.device)
                    
                    # Since we didn't normalize input at dataloading phase
                    data.sub_(self.dmean[None,:,None,None]).div_(self.dstd[None,:,None,None])
                    
                    output = self.model(data)
                    loss = self.val_criterion(output, label)

                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, epoch=epoch, mode = 'valid')
                    self.writer.add_scalar({'loss': loss.item()})
                    self.val_loss_list.append(loss.item())
                    total_val_loss += loss.item()
                    total_val_metrics += self._eval_metrics(output, label)
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        
        val_acc = (total_val_metrics / len(self.valid_data_loader)).tolist()[0]
        if val_acc > self.val_acc:
            self.val_acc = val_acc
            self.new_best_val = True
            self.writer.add_scalar({'Best val acc': self.val_acc}, epoch = epoch)
        else:
            self.new_best_val = False

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def _test_epoch(self, epoch):
        """
        Test after training an epoch
        :return: A log that contains information about test
        Note:
            The Test metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_test_loss = 0
        total_test_metrics = np.zeros(len(self.metrics))
        
        with torch.no_grad():
            with tqdm(self.test_data_loader) as progress:
                for batch_idx, (data, label) in enumerate(progress):
                    progress.set_description_str(f'Test epoch {epoch}')
                    data, label = data.to(self.device), label.to(self.device)
                    
                    # Since we didn't normalize input at dataloading phase
                    data.sub_(self.dmean[None,:,None,None]).div_(self.dstd[None,:,None,None])
                    
                    output = self.model(data)
                    
                    loss = self.val_criterion(output, label)

                    self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, epoch=epoch, mode = 'test')
                    self.writer.add_scalar({'loss': loss.item()})
                    self.test_loss_list.append(loss.item())
                    total_test_loss += loss.item()
                    total_test_metrics += self._eval_metrics(output, label)
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        top_1_acc = (total_test_metrics / len(self.test_data_loader)).tolist()[0]
        if self.new_best_val:
            self.test_val_acc = top_1_acc
            self.writer.add_scalar({'Test acc with best val': top_1_acc}, epoch = epoch)
        self.writer.add_scalar({'Top-1': top_1_acc}, epoch = epoch)
        self.writer.add_scalar({'Top-5': (total_test_metrics / len(self.test_data_loader)).tolist()[1]}, epoch = epoch)

        return {
            'test_loss': total_test_loss / len(self.test_data_loader),
            'test_metrics': (total_test_metrics / len(self.test_data_loader)).tolist()
        }


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
