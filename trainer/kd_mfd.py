from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from utils import get_accuracy
from trainer.kd_hinton import Trainer as hinton_Trainer
from trainer.loss_utils import compute_hinton_loss


class Trainer(hinton_Trainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.lambh = args.lambh
        self.lambf = args.lambf
        self.sigma = args.sigma
        self.kernel = args.kernel
        self.jointfeature = args.jointfeature

    def train(self, train_loader, test_loader, epochs):

        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups

        distiller = MMDLoss(w_m=self.lambf, sigma=self.sigma,
                            num_classes=num_classes, num_groups=num_groups, kernel=self.kernel)

        for epoch in range(self.epochs):
            self._train_epoch(epoch, train_loader, self.model, self.teacher, distiller=distiller)
            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deopp = self.evaluate(self.model, test_loader, self.criterion)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEopp {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_deopp, (eval_end_time - eval_start_time)))

            if self.scheduler != None:
                self.scheduler.step(eval_loss)

        print('Training Finished!')

    def _train_epoch(self, epoch, train_loader, model, teacher, distiller=None):
        model.train()
        teacher.eval()

        running_acc = 0.0
        running_loss = 0.0
        batch_start_time = time.time()

        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data
            labels = targets

            if self.cuda:
                inputs = inputs.cuda(self.device)
                labels = labels.cuda(self.device)
                groups = groups.long().cuda(self.device)
            t_inputs = inputs.to(self.t_device)

            outputs = model(inputs, get_inter=True)
            stu_logits = outputs[-1]

            t_outputs = teacher(t_inputs, get_inter=True)
            tea_logits = t_outputs[-1]

            kd_loss = compute_hinton_loss(stu_logits, t_outputs=tea_logits,
                                          kd_temp=self.kd_temp, device=self.device) if self.lambh != 0 else 0

            loss = self.criterion(stu_logits, labels)
            loss = loss + self.lambh * kd_loss


            f_s = outputs[-2]
            f_t = t_outputs[-2]
            mmd_loss = distiller.forward(f_s, f_t, groups=groups, labels=labels, jointfeature=self.jointfeature)

            loss = loss + mmd_loss
            running_loss += loss.item()
            running_acc += get_accuracy(stu_logits, labels)

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

        if not self.no_annealing:
            self.lambh = self.lambh - 3 / (self.epochs - 1)


class MMDLoss(nn.Module):
    def __init__(self, w_m, sigma, num_groups, num_classes, kernel):
        super(MMDLoss, self).__init__()
        self.w_m = w_m
        self.sigma = sigma
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.kernel = kernel

    def forward(self, f_s, f_t, groups, labels, jointfeature=False):
        if self.kernel == 'poly':
            student = F.normalize(f_s.view(f_s.shape[0], -1), dim=1)
            teacher = F.normalize(f_t.view(f_t.shape[0], -1), dim=1).detach()
        else:
            student = f_s.view(f_s.shape[0], -1)
            teacher = f_t.view(f_t.shape[0], -1).detach()

        mmd_loss = 0

        if jointfeature:
            K_TS, sigma_avg = self.pdist(teacher, student,
                              sigma_base=self.sigma, kernel=self.kernel)
            K_TT, _ = self.pdist(teacher, teacher, sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)
            K_SS, _ = self.pdist(student, student,
                              sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

            mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()

        else:
            with torch.no_grad():
                _, sigma_avg = self.pdist(teacher, student, sigma_base=self.sigma, kernel=self.kernel)

            for c in range(self.num_classes):
                if len(teacher[labels==c]) == 0:
                    continue
                for g in range(self.num_groups):
                    if len(student[(labels==c) * (groups == g)]) == 0:
                        continue
                    K_TS, _ = self.pdist(teacher[labels == c], student[(labels == c) * (groups == g)],
                                                 sigma_base=self.sigma, sigma_avg=sigma_avg,  kernel=self.kernel)
                    K_SS, _ = self.pdist(student[(labels == c) * (groups == g)], student[(labels == c) * (groups == g)],
                                         sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

                    K_TT, _ = self.pdist(teacher[labels == c], teacher[labels == c], sigma_base=self.sigma,
                                         sigma_avg=sigma_avg, kernel=self.kernel)

                    mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()

        loss = (1/2) * self.w_m * mmd_loss

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
                res = torch.exp(-res / (2*(sigma_base)*sigma_avg))
            elif kernel == 'poly':
                res = torch.matmul(e1, e2.t()).pow(2)

        return res, sigma_avg
