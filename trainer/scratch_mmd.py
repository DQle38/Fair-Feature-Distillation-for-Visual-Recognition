from __future__ import print_function

import time
import numpy as np
from utils import get_accuracy
import trainer
import torch
import torch.nn as nn
import torch.nn.functional as F


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)

        self.lambf = args.lambf
        self.sigma = args.sigma
        self.kernel = args.kernel

    def train(self, train_loader, test_loader, epochs):
        model = self.model
        model.train()

        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups

        distiller = MMDLoss(w_m=self.lambf, sigma=self.sigma,
                            num_classes=num_classes, num_groups=num_groups, kernel=self.kernel)

        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, model, distiller=distiller)

            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deopp = self.evaluate(model, test_loader, self.criterion)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEopp {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_deopp, (eval_end_time - eval_start_time)))

            if self.scheduler != None and 'Multi' not in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()

        print('Training Finished!')

    def _train_epoch(self, epoch, train_loader, model, distiller):

        model.train()

        running_acc = 0.0
        running_loss = 0.0
        batch_start_time = time.time()
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data

            labels = targets

            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.long().cuda(device=self.device)

            outputs = model(inputs, get_inter=True)
            f_s = outputs[-2]
            loss = self.criterion(outputs[-1], labels)

            mmd_loss = distiller.forward(f_s, groups=groups, labels=labels)
            loss = loss + mmd_loss

            running_loss += loss.item()
            running_acc += get_accuracy(outputs[-1], labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.term == self.term - 1:  # print every self.term mini-batches
                avg_batch_time = time.time() - batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i + 1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time / self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()


class MMDLoss(nn.Module):
    def __init__(self, w_m, sigma, num_groups, num_classes, kernel):
        super(MMDLoss, self).__init__()
        self.w_m = w_m
        self.sigma = sigma
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.kernel = kernel

    def forward(self, f_s, groups, labels):
        if self.kernel == 'poly':
            student = F.normalize(f_s.view(f_s.shape[0], -1), dim=1)
        else:
            student = f_s.view(f_s.shape[0], -1)

        mmd_loss = 0

        for c in range(self.num_classes):

            target_joint = student[labels == c].clone().detach()

            for g in range(self.num_groups):
                if len(student[(labels == c) * (groups == g)]) == 0:
                    continue

                K_SSg, sigma_avg = self.pdist(target_joint, student[(labels == c) * (groups == g)],
                                              sigma_base=self.sigma, kernel=self.kernel)

                K_SgSg, _ = self.pdist(student[(labels==c) * (groups==g)], student[(labels==c) * (groups==g)],
                                       sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

                K_SS, _ = self.pdist(target_joint, target_joint,
                                     sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

                mmd_loss += torch.clamp(K_SS.mean() + K_SgSg.mean() - 2 * K_SSg.mean(), 0.0, np.inf).mean()

        loss = self.w_m * mmd_loss / (2*self.num_groups)

        return loss

    @staticmethod
    def pdist(e1, e2, eps=1e-12, kernel='rbf', sigma_base=1.0, sigma_avg=None):
        if len(e1) == 0 or len(e2) == 0:
            res = torch.zeros(1)
        else:
            if kernel == 'rbf':
                e1_square = e1.pow(2).sum(dim=1)
                e2_square = e2.pow(2).sum(dim=1)
                prod = e1 @ e2.t()
                res = (e1_square.unsqueeze(1) + e2_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
                res = res.clone()

                sigma_avg = res.mean().detach() if sigma_avg is None else sigma_avg
                res = torch.exp(-res / (2*(sigma_base**2)*sigma_avg))
            elif kernel == 'poly':
                res = torch.matmul(e1, e2.t()).pow(2)
        return res, sigma_avg
