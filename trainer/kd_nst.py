from __future__ import print_function

import torch.nn.functional as F
import torch.nn as nn
import time
from utils import get_accuracy
import trainer


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.model_type = args.model
        self.lambf = args.lambf
        self.kernel = args.kernel

    def train(self, train_loader, test_loader, epochs):

        distiller = NST(lamb=self.lambf)

        for epoch in range(self.epochs):

            self._train_epoch(epoch, train_loader, self.model, self.teacher, distiller)

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

    def _train_epoch(self, epoch, train_loader, model, teacher, distiller):
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
            t_inputs = inputs.to(self.t_device)

            outputs = model(inputs, get_inter=True)
            stu_logits = outputs[-1]
            f_s = outputs[-2]

            t_outputs = teacher(t_inputs, get_inter=True)
            f_t = t_outputs[-2]

            loss = self.criterion(stu_logits, labels)
            mmd_loss = distiller.forward(f_s, f_t)

            loss = loss + mmd_loss

            running_loss += loss.item()
            running_acc += get_accuracy(stu_logits, labels)

            # zero the parameter gradients + backward + optimize
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


class NST(nn.Module):
    def __init__(self, lamb):
        super(NST, self).__init__()
        self.lamb = lamb

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), fm_s.size(1), -1)
        fm_s = F.normalize(fm_s, dim=2)

        fm_t = fm_t.view(fm_t.size(0), fm_t.size(1), -1)
        fm_t = F.normalize(fm_t, dim=2)

        loss = self.poly_kernel(fm_t, fm_t).mean() + self.poly_kernel(fm_s, fm_s).mean() \
               - 2 * self.poly_kernel(fm_t, fm_s).mean()

        loss = self.lamb * loss
        return loss

    def poly_kernel(self, fm1, fm2):
        fm1 = fm1.unsqueeze(1)
        fm2 = fm2.unsqueeze(2)
        out = (fm1 * fm2).sum(-1).pow(2)

        return out
