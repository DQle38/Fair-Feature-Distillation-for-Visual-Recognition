from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import time

from utils import get_accuracy
from networks.mlp import MLP
from trainer.loss_utils import compute_hinton_loss, compute_feature_loss
import trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, teacher, **kwargs):
        super().__init__(args=args, **kwargs)
        self.lambh = args.lambh
        self.lambf = args.lambf
        self.kd_temp = args.kd_temp
        self.teacher = teacher

        self.adv_lambda = args.adv_lambda
        self.adv_lr = args.eta
        self.no_annealing = args.no_annealing

    def train(self, train_loader, test_loader, epochs):
        model = self.model
        model.train()
        num_groups = train_loader.dataset.num_groups
        num_classes = train_loader.dataset.num_classes
        if self.teacher is not None:
            self.teacher.eval()
        self._init_adversary(num_groups, num_classes)
        sa_clf_list = self.sa_clf_list

        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, model, sa_clf_list, self.teacher)

            eval_start_time = time.time()
            eval_loss, eval_acc, eval_adv_loss, eval_adv_acc, eval_deopp, eval_adv_loss_list = \
                self.evaluate(model, sa_clf_list, test_loader, self.criterion, self.adv_criterion)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test Adv Acc: {:.2f} Test DEopp {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_adv_acc, eval_deopp, (eval_end_time - eval_start_time)))

            if self.scheduler != None:
                self.scheduler.step(eval_loss)
            if len(self.adv_scheduler_list) != 0:
                for c in range(num_classes):
                    self.adv_scheduler_list[c].step(eval_adv_loss_list[c])

        print('Training Finished!')

    def _train_epoch(self, epoch, train_loader, model, sa_clf, teacher=None):
        num_classes = train_loader.dataset.num_classes

        model.train()
        if teacher is not None:
            teacher.eval()

        running_acc = 0.0
        running_loss = 0.0
        batch_start_time = time.time()

        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data
            labels = targets
            groups = groups.long()

            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.cuda(device=self.device)

            kd_loss = 0
            feature_loss = 0
            if teacher is not None:
                t_inputs = inputs.to(self.t_device)
                feature_loss, outputs, tea_logits, _, _ = compute_feature_loss(inputs, t_inputs, model, teacher,
                                                                               device=self.device)
                kd_loss = compute_hinton_loss(outputs, t_outputs=tea_logits,
                                              kd_temp=self.kd_temp, device=self.device) if self.lambh != 0 else 0
            else:
                outputs = model(inputs)

            adv_loss = 0
            for c in range(num_classes):
                if sum(labels == c) == 0:
                    continue
                adv_inputs = outputs[labels == c].clone()
                adv_preds = sa_clf[c](adv_inputs)
                adv_loss += self.adv_criterion(adv_preds, groups[labels==c])

            loss = self.criterion(outputs, labels)
            loss = loss + self.lambh * kd_loss
            loss = loss + self.lambf * feature_loss

            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)

            self.optimizer.zero_grad()
            for c in range(num_classes):
                self.adv_optimizer_list[c].zero_grad()

            loss = loss + adv_loss
            loss.backward()

            self.optimizer.step()
            for c in range(num_classes):
                self.adv_optimizer_list[c].step()

            if i % self.term == self.term - 1:  # print every self.term mini-batches
                avg_batch_time = time.time() - batch_start_time
                print_statement = '[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} [{:.2f} s/batch]'\
                    .format(epoch + 1, self.epochs, i + 1, self.method, running_loss / self.term, running_acc / self.term,
                            avg_batch_time / self.term)
                print(print_statement)

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()

        if not self.no_annealing and teacher is not None:
            self.lambh = self.lambh - 3/(self.epochs-1)

    def evaluate(self, model, adversary, loader, criterion, adv_criterion, device=None):
        model.eval()
        num_groups = loader.dataset.num_groups
        num_classes = loader.dataset.num_classes
        device = self.device if device is None else device
        eval_acc = 0
        eval_adv_acc = 0
        eval_loss = 0
        eval_adv_loss = 0
        eval_adv_loss_list = torch.zeros(num_classes)
        eval_eopp_list = torch.zeros(num_groups, num_classes).cuda(device)
        eval_data_count = torch.zeros(num_groups, num_classes).cuda(device)

        if 'Custom' in type(loader).__name__:
            loader = loader.generate()
        with torch.no_grad():
            for j, eval_data in enumerate(loader):
                # Get the inputs
                inputs, _, groups, classes, _ = eval_data
                #
                labels = classes
                groups = groups.long()
                if self.cuda:
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    groups = groups.cuda(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                eval_loss += loss.item()
                eval_acc += get_accuracy(outputs, labels)
                preds = torch.argmax(outputs, 1)
                acc = (preds == labels).float().squeeze()
                for g in range(num_groups):
                    for l in range(num_classes):
                        eval_eopp_list[g, l] += acc[(groups == g) * (labels == l)].sum()
                        eval_data_count[g, l] += torch.sum((groups == g) * (labels == l))

                for c in range(num_classes):
                    if sum(labels==c)==0:
                        continue
                    adv_preds = adversary[c](outputs[labels==c])
                    adv_loss = adv_criterion(adv_preds, groups[labels==c])
                    eval_adv_loss += adv_loss.item()
                    eval_adv_loss_list[c] += adv_loss.item()
                    # print(c, adv_preds.shape)
                    eval_adv_acc += get_accuracy(adv_preds, groups[labels==c])

            eval_loss = eval_loss / (j+1)
            eval_acc = eval_acc / (j+1)
            eval_adv_loss = eval_adv_loss / ((j+1) * num_classes)
            eval_adv_loss_list = eval_adv_loss_list / (j+1)
            eval_adv_acc = eval_adv_acc / ((j+1) * num_classes)
            eval_eopp_list = eval_eopp_list / eval_data_count
            eval_max_eopp = torch.max(eval_eopp_list, dim=0)[0] - torch.min(eval_eopp_list, dim=0)[0]
            eval_max_eopp = torch.max(eval_max_eopp).item()
        model.train()
        return eval_loss, eval_acc, eval_adv_loss, eval_adv_acc, eval_max_eopp, eval_adv_loss_list

    def _init_adversary(self, num_groups, num_classes):
        self.model.eval()
        self.sa_clf_list = []
        self.adv_optimizer_list = []
        self.adv_scheduler_list = []
        for _ in range(num_classes):
            sa_clf = MLP(feature_size=num_classes, hidden_dim=32, num_class=num_groups, num_layer=2,
                         adv=True, adv_lambda=self.adv_lambda)
            if self.cuda:
                sa_clf.cuda(device=self.device)
            sa_clf.train()
            self.sa_clf_list.append(sa_clf)
            adv_optimizer = optim.Adam(sa_clf.parameters(), lr=self.adv_lr)
            self.adv_optimizer_list.append(adv_optimizer)
            self.adv_scheduler_list.append(ReduceLROnPlateau(adv_optimizer))

        self.adv_criterion = nn.CrossEntropyLoss()
