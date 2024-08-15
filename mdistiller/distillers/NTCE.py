import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def shrink_margin(y_t, y_s, target, factors):
    # get max class value and its index
    max_class_value, max_class_index = torch.max(y_t, dim=1)
    # get second class value and its index
    second_class_value, second_class_index = torch.topk(y_t, k=2, dim=1, largest=True, sorted=True)
    second_class_value = second_class_value[:, 1]
    second_class_index = second_class_index[:, 1]    
    # get the shrink value, i.e. max_one - second_one
    shrink_value = max_class_value - second_class_value
    shrink_value_list = [shrink_value * f for f in factors]
    # update the max class of the teacher logits
    max_class_index = torch.zeros_like(y_t).scatter_(1, max_class_index.unsqueeze(1), 1)
    # assert shrink_value.shape == (64, 1), shrink_value.shape
    # assert max_class_index.shape == (64, 1), max_class_index.shape
    y_t_opt_list = [y_t - shrink_value_list[i].unsqueeze(1) * max_class_index for i in range(len(factors))]
    
    # calculate the margin based on the std and the shrink of the sample
    margin_list = [s_value.unsqueeze(1) for s_value in shrink_value_list]
    # update the target class of the student logits
    target_class_index = torch.zeros_like(y_s).scatter_(1, target.unsqueeze(1), 1)
    y_s_opt_list = [y_s - margin * target_class_index for margin in margin_list]

    return y_t_opt_list, y_s_opt_list


def kd_loss(y_s, y_t, temperature):
    p_s = F.log_softmax(y_s/temperature, dim=1)
    p_t = F.softmax(y_t/temperature, dim=1)
    loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (temperature**2)
    
    return loss


class NTCE(Distiller):
    """NTCE-KD: Non-Target-Class-Enhanced Knowledge Distillation."""

    def __init__(self, student, teacher, cfg):
        super(NTCE, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.NTCE.CE_WEIGHT
        self.kd_loss_weight = cfg.NTCE.KD_WEIGHT
        self.temperature = cfg.NTCE.T
        self.warmup = cfg.NTCE.WARMUP
        self.isaug = cfg.NTCE.ISAUG
        self.factors = cfg.NTCE.FACTORS

    def get_learnable_parameters(self):
        return super().get_learnable_parameters()

    def forward_train(self, image, image_strong=None, target=None, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        
        # get logits_teacher and logits_student
        logits_teacher_opt_list, logits_student_opt_list = \
            shrink_margin(logits_teacher, logits_student, target, self.factors)
        
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target) 

        loss_dm = [kd_loss(
            logits_student_opt,
            logits_teacher_opt.detach(),
            self.temperature)
            for logits_student_opt, logits_teacher_opt in zip(logits_student_opt_list, logits_teacher_opt_list)]

        if self.isaug:
            logits_student_strong, _ = self.student(image_strong)
            with torch.no_grad():
                logits_teacher_strong, _ = self.teacher(image_strong)

            logits_teacher_strong_opt_list, logits_student_strong_opt_list, _ = \
                shrink_margin(logits_teacher_strong, logits_student_strong, target, self.factors)        
        
            # losses
            loss_ce_strong = self.ce_loss_weight * F.cross_entropy(logits_student_strong, target)
            
            loss_dm_strong = [kd_loss(
                logits_student_strong_opt,
                logits_teacher_strong_opt.detach(),
                self.temperature,)
                for logits_student_strong_opt, logits_teacher_strong_opt in zip(logits_student_strong_opt_list, logits_teacher_strong_opt_list)]

        loss_ce = (loss_ce + loss_ce_strong) / 2 if self.isaug else loss_ce
        loss_dm = (sum(loss_dm) + sum(loss_dm_strong)) / 2 * len(loss_dm) if self.isaug else sum(loss_dm) / len(loss_dm)
        loss_kd = min(kwargs["epoch"] / self.warmup, 1.0) * self.kd_loss_weight * (loss_dm)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
