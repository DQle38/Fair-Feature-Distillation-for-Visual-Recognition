from __future__ import print_function
import time
from utils import get_accuracy
from trainer.loss_utils import compute_at_loss
import trainer


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)

        self.model_type = args.model
        self.for_cifar = True if args.model == 'cifar_net' else False
        self.lambf = args.lambf

    def train(self, train_loader, test_loader, epochs):

        self.model.train()
        self.teacher.eval()
        
        for epoch in range(self.epochs):
            # train during one epoch
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

    def _train_epoch(self, epoch, train_loader, model, teacher):
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

            attention_loss, stu_outputs, tea_outputs, _, _ = compute_at_loss(inputs, t_inputs, model, teacher,
                                                                             self.device, for_cifar=self.for_cifar)
            loss = self.criterion(stu_outputs, labels)
            loss = loss + self.lambf * attention_loss

            running_loss += loss.item()
            running_acc += get_accuracy(stu_outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.term == self.term-1: # print every self.term mini-batches
                avg_batch_time = time.time() - batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i+1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time/self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()

