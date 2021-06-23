# Fair-Feature-Distillation-for-Visual-Recognition
Official implementation of paper 'Fair Feature Distillation for Visual Recognition'

## **Execution Details**

### Requirements

- Python 3
- GPU Titan XP / Pytorch 1.6 / CUDA 10.1

#### 1) Download dataset

- UTKFACE : [link](https://susanqq.github.io/UTKFace/) (We used Aligned&Cropped Faces from the site)
- CelebA : link

#### 2) Execution command
You should first train a scratch model used as a teacher.
```
# Cifar10
$ python3 ./main.py --method scratch --dataset cifar10 --model cifar_net --epochs 50 --img-size 32 --batch-size 128 --optimizer Adam --lr 0.001 --date 210525

# UTKFACE
$ python3 ./main.py --method scratch --dataset utkface --model resnet18 --epochs 50 --img-size 176 --batch-size 128 --optimizer Adam --lr 0.001 --date 210525

# CelebA
$ python3 ./main.py --method scratch --dataset celeba --model shufflenet --epochs 50 --img-size 176 --batch-size 128 --optimizer Adam --lr 0.001 --date 210525
```

Then, using the saved teacher model, you can train a student model via MFD algorithm.
```
# Cifar10
$ python3 ./main.py --method kd_mfd --dataset cifar10 --model cifar_net --epochs 50 --labelwise --lambf 3 --lambh 0 --no-annealing --img-size 32 --batch-size 128 --optimizer Adam --lr 0.001 --teacher-path trained_models/210525/cifar10/scratch/cifar_net_seed0_epochs50_bs128_lr0.001.pt

# UTKFACE
$ python3 ./main.py --method kd_mfd --dataset utkface --model resnet18 --epochs 50 --labelwise --lambf 3 --lambh 0 --no-annealing --img-size 176 --batch-size 128 --optimizer Adam --lr 0.001 --teacher-path trained_models/210525/utkface/scratch/resnet18_seed0_epochs50_bs128_lr0.001.pt

# CelebA
$ python3 ./main.py --method kd_mfd --dataset celeba --model shufflenet --epochs 50 --labelwise --lambf 7 --lambh 0 --no-annealing --img-size 176 --batch-size 128 --optimizer Adam --lr 0.001 --teacher-path trained_models/210525/celeba/scratch/shufflenet_seed0_epochs50_bs128_lr0.001.pt
```

#### Notes

The all datasets can be downloaded in link.

