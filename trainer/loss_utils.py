import torch.nn as nn
import torch.nn.functional as F


def mse(inputs, targets):
    return (inputs - targets).pow(2).mean()


def compute_feature_loss(inputs, t_inputs, student, teacher, device=0):
    stu_outputs = student(inputs, get_inter=True)
    f_s = stu_outputs[-2]
    f_s = f_s.view(f_s.shape[0], -1)
    stu_logits = stu_outputs[-1]

    tea_outputs = teacher(t_inputs, get_inter=True)
    f_t = tea_outputs[-2].to(device)
    f_t = f_t.view(f_t.shape[0], -1).detach()

    tea_logits = tea_outputs[-1]

    fitnet_loss = (1 / 2) * (mse(f_s, f_t))

    return fitnet_loss, stu_logits, tea_logits, f_s, f_t


def compute_hinton_loss(outputs, t_outputs=None, teacher=None, t_inputs=None, kd_temp=3, device=0):
    if t_outputs is None:
        if (t_inputs is not None and teacher is not None):
            t_outputs = teacher(t_inputs)
        else:
            Exception('Nothing is given to compute hinton loss')

    soft_label = F.softmax(t_outputs / kd_temp, dim=1).to(device).detach()
    kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / kd_temp, dim=1),
                                                  soft_label) * (kd_temp * kd_temp)

    return kd_loss


def compute_at_loss(inputs, t_inputs, student, teacher, device=0, for_cifar=False):
    stu_outputs = student(inputs, get_inter=True) if not for_cifar else student(inputs, get_inter=True, before_fc=True)
    stu_logits = stu_outputs[-1]
    f_s = stu_outputs[-2]

    tea_outputs = teacher(t_inputs, get_inter=True) if not for_cifar else teacher(inputs, get_inter=True, before_fc=True)
    tea_logits = tea_outputs[-1].to(device)
    f_t = tea_outputs[-2].to(device)
    attention_loss = (1 / 2) * (at_loss(f_s, f_t))
    return attention_loss, stu_logits, tea_logits, f_s, f_t


def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()
