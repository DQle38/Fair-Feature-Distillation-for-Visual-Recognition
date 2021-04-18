from __future__ import print_function

import time
from utils import get_accuracy
from trainer.kd_hinton import Trainer as hinton_Trainer
from trainer.loss_utils import compute_feature_loss, compute_hinton_loss


class Trainer(hinton_Trainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.model_type = args.model

        self.fitnet_simul = args.fitnet_simul

    def train(self, train_loader, test_loader, epochs):

        if not self.fitnet_simul:
            for epoch in range(int(self.epochs/2)):

                self._train_epoch_hint(epoch, train_loader, self.model, self.teacher)

            print('Hint Training Finished!')
            self.save_model(self.save_dir, self.log_name + '_hint')

        for epoch in range(self.epochs):
            self._train_epoch(epoch, train_loader, self.model, self.teacher)
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

    def _train_epoch_hint(self, epoch, train_loader, model, teacher):
        model.train()
        teacher.eval()

        running_loss = 0.0
        avg_batch_time = 0.0

        for i, data in enumerate(train_loader):
            batch_start_time = time.time()
            # Get the inputs
            inputs, _, groups, labels, _ = data

            if self.cuda:
                inputs = inputs.cuda(self.device)
            t_inputs = inputs.to(self.t_device)

            fitnet_loss, _, _, _, _ = compute_feature_loss(inputs, t_inputs, model, teacher, device=self.device)
            running_loss += fitnet_loss.item()

            self.optimizer.zero_grad()
            fitnet_loss.backward()
            self.optimizer.step()

            batch_end_time = time.time()
            avg_batch_time += batch_end_time - batch_start_time

            if i % self.term == self.term-1:  # print every self.term mini-batches
                train_loss = running_loss / self.term
                print('[{}/{}, {:5d}] Method: {} FitNet Hint Train Loss: {:.3f} [{:.2f} s/batch]'.format
                      (epoch + 1, int(self.epochs/2), i + 1, self.method, train_loss, avg_batch_time / self.term))

                running_loss = 0.0
                avg_batch_time = 0.0

    def _train_epoch(self, epoch, train_loader, model, teacher, distiller=None):
        model.train()
        teacher.eval()

        running_acc = 0.0
        running_loss = 0.0
        batch_start_time = time.time()

        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, _, targets, _ = data
            labels = targets

            if self.cuda:
                inputs = inputs.cuda(self.device)
                labels = labels.cuda(self.device)
            t_inputs = inputs.to(self.t_device)

            if self.fitnet_simul:
                feature_loss, stu_logits, tea_logits, _, _ = compute_feature_loss(inputs, t_inputs, model, teacher,
                                                                                  device=self.device)
                kd_loss = compute_hinton_loss(stu_logits, t_outputs=tea_logits,
                                              kd_temp=self.kd_temp, device=self.device)
            else:
                stu_logits = model(inputs)
                kd_loss = compute_hinton_loss(stu_logits, t_inputs=t_inputs, teacher=teacher,
                                              kd_temp=self.kd_temp, device=self.device) if self.lambh != 0 else 0
                feature_loss = 0

            loss = self.criterion(stu_logits, labels)

            loss = loss + self.lambh * kd_loss
            loss = loss + feature_loss if self.fitnet_simul else loss

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
            self.lambh = self.lambh - 3/(self.epochs-1)

