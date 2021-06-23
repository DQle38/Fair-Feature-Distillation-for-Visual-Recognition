import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networks
import data_handler
import trainer
from utils import check_log_dir, make_log_name, set_seed

from arguments import get_args
import time
import os 

args = get_args()


def main():

    torch.backends.cudnn.enabled = True

    seed = args.seed
    set_seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    log_name = make_log_name(args)
    dataset = args.dataset
    save_dir = os.path.join(args.save_dir, args.date, dataset, args.method)
    log_dir = os.path.join(args.log_dir, args.date, dataset, args.method)
    check_log_dir(save_dir)
    check_log_dir(log_dir)    
    ########################## get dataloader ################################
    
    tmp = data_handler.DataloaderFactory.get_dataloader(args.dataset, img_size=args.img_size,
                                                        batch_size=args.batch_size, seed=args.seed,
                                                        num_workers=args.num_workers,
                                                        target=args.target,
                                                        skew_ratio=args.skew_ratio,
                                                        labelwise=args.labelwise
                                                        )
    num_classes, num_groups, train_loader, test_loader = tmp
    
    ########################## get model ##################################

    model = networks.ModelFactory.get_model(args.model, num_classes, args.img_size, pretrained=args.pretrained)

    if args.parallel:
        model = nn.DataParallel(model)

    model.cuda('cuda:{}'.format(args.device))

    if args.modelpath is not None:
        model.load_state_dict(torch.load(args.modelpath))

    teacher = None
    if (args.method.startswith('kd') or args.teacher_path is not None) and args.mode != 'eval':
        teacher = networks.ModelFactory.get_model(args.model, train_loader.dataset.num_classes, args.img_size)
        if args.parallel:
            teacher = nn.DataParallel(teacher)
        teacher.load_state_dict(torch.load(args.teacher_path))
        teacher.cuda('cuda:{}'.format(args.t_device))

    ########################## get trainer ##################################

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, )
    elif 'SGD' in args.optimizer:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    trainer_ = trainer.TrainerFactory.get_trainer(args.method, model=model, args=args,
                                                  optimizer=optimizer, teacher=teacher)

    ####################### start training or evaluating ####################
    
    if args.mode == 'train':
        start_t = time.time()
        trainer_.train(train_loader, test_loader, args.epochs)
        end_t = time.time()
        train_t = int((end_t - start_t)/60)  # to minutes
        print('Training Time : {} hours {} minutes'.format(int(train_t/60), (train_t % 60)))
        trainer_.save_model(save_dir, log_name)
    
    else:
        print('Evaluation ----------------')
        model_to_load = args.modelpath
        trainer_.model.load_state_dict(torch.load(model_to_load))
        print('Trained model loaded successfully')

    if args.evalset == 'all':
        trainer_.compute_confusion_matix('train', train_loader.dataset.num_classes, train_loader, log_dir, log_name)
        trainer_.compute_confusion_matix('test', test_loader.dataset.num_classes, test_loader, log_dir, log_name)

    elif args.evalset == 'train':
        trainer_.compute_confusion_matix('train', train_loader.dataset.num_classes, train_loader, log_dir, log_name)
    else:
        trainer_.compute_confusion_matix('test', test_loader.dataset.num_classes, test_loader, log_dir, log_name)

    print('Done!')


if __name__ == '__main__':
    main()
