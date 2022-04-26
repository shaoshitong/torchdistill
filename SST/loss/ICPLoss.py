from torchdistill.models.util import wrap_if_distributed, load_module_ckpt, save_module_ckpt, redesign_model
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import adaptive_avg_pool2d, adaptive_max_pool2d, normalize, cosine_similarity
from torchdistill.losses.single import register_single_loss
import einops


class KLloss(nn.Module):
    def __init__(self, negative_weight, positive_weight, temperature=4.0):
        super(KLloss, self).__init__()
        self.kl = nn.KLDivLoss(reduction="none")
        self.negative_weight = negative_weight
        self.positive_weight = positive_weight
        self.temperature = temperature

    def generate_weight(self, pred, target, gold):
        one_hot = torch.full_like(pred, fill_value=self.negative_weight).to(pred.device)
        one_hot.scatter_(dim=1, index=gold.argmax(1).unsqueeze(1), value=self.positive_weight)  # 0.9
        return one_hot

    def forward(self, pred, target, gold):  # b,n b,n
        kl_loss = self.kl(torch.log_softmax(pred / self.temperature, dim=1),
                          target) * (self.temperature)
        weight = self.generate_weight(pred, target, gold)
        return (kl_loss * weight).mean()


@register_single_loss
class AuxICPLoss(nn.Module):
    def __init__(self, module_path='policy_module', module_io='output',
                 negative_loss_weight=[1.0, 0.5, 0.001], positive_loss_weight=[1.0, 0.5, 0.001], **kwargs):
        super().__init__()
        self.module_path = module_path
        self.module_io = module_io
        self.type = type
        self.positive_loss_weight = positive_loss_weight
        self.negative_loss_weight = negative_loss_weight
        self.identity_kl_loss = KLloss(negative_loss_weight[0], positive_loss_weight[0], 1)
        self.classes_kl_loss = KLloss(negative_loss_weight[1], positive_loss_weight[1], 1)
        self.policy_kl_loss = KLloss(negative_loss_weight[2], positive_loss_weight[2], 1)
        self.iter=0.
        print(f"p loss weight is {positive_loss_weight},{negative_loss_weight}")

    def forward(self, student_io_dict, teacher_io_dict, target, *args, **kwargs):
        icp_module_outputs = teacher_io_dict[self.module_path][self.module_io]
        output_identity, output_classes, output_policy = icp_module_outputs
        b, p, l = target.shape
        target = target.view(-1, l)
        target_identity, target_classes = F.one_hot(target[:, 0], output_identity.shape[1]).cuda(), F.one_hot(
            target[:, 1], output_classes.shape[1]).cuda()
        target_identity, target_classes = target_identity.float(), target_classes.float()
        identity_loss = self.identity_kl_loss(output_identity, target_identity, target_identity)
        classes_loss = self.classes_kl_loss(output_classes, target_classes, target_classes)
        policy_loss = 0.
        indices = torch.arange(1,b*2,2)
        indices2= torch.arange(0,b*2,2)
        for i in range(target.shape[1] - 2):
            target_policy_one = F.one_hot(target[indices][:, i + 2], 2).cuda().float()
            policy_loss += self.policy_kl_loss(output_policy[indices][:, 2 * i:2 * (i + 1)]-output_policy[indices2][:, 2 * i:2 * (i + 1)].detach(), target_policy_one,
                                               target_policy_one)
        return identity_loss + classes_loss + policy_loss


@register_single_loss
class ICPLoss(nn.Module):
    def __init__(self, student_linear_module_path, teacher_linear_module_path, student_policy_module_path,
                 teacher_policy_module_path, kl_temp,
                 student_linear_module_io='output', teacher_linear_module_io='output',
                 student_policy_module_io='output', teacher_policy_module_io='output',
                 loss_weights=None, kd_and_ce_weight=[1, 1], negative_loss_weight=[1.0, 0.5, 0.001],
                 positive_loss_weight=[1.0, 0.5, 0.001], temperature=1.0, adnamic_weight=True, **kwargs):
        super().__init__()
        self.loss_weights = [1.0, 1.0, 1.0] if loss_weights is None else loss_weights
        print("ce,kd,policy loss weight is", self.loss_weights)
        print("positive and negative loss weight is", positive_loss_weight, negative_loss_weight)
        print("policy's kd and ce weight is ", kd_and_ce_weight)
        self.kl_temp = kl_temp
        self.positive_loss_weight = positive_loss_weight
        self.negative_loss_weight = negative_loss_weight
        self.kd_and_ce_weight = kd_and_ce_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="mean")
        self.kldiv_loss = nn.KLDivLoss(reduction="mean")
        self.student_linear_module_path = student_linear_module_path
        self.student_linear_module_io = student_linear_module_io
        self.teacher_linear_module_path = teacher_linear_module_path
        self.teacher_linear_module_io = teacher_linear_module_io
        self.student_policy_module_path = student_policy_module_path
        self.student_policy_module_io = student_policy_module_io
        self.teacher_policy_module_path = teacher_policy_module_path
        self.teacher_policy_module_io = teacher_policy_module_io
        self.identity_kl_loss = KLloss(negative_loss_weight[0], positive_loss_weight[0], temperature)
        self.classes_kl_loss = KLloss(negative_loss_weight[1], positive_loss_weight[1], temperature)
        self.policy_kl_loss = KLloss(negative_loss_weight[2], positive_loss_weight[2], temperature)
        self.identity_ce_loss = KLloss(negative_loss_weight[0], positive_loss_weight[0], 1)
        self.classes_ce_loss = KLloss(negative_loss_weight[1], positive_loss_weight[1], 1)
        self.policy_ce_loss = KLloss(negative_loss_weight[2], positive_loss_weight[2], 1)
        self.adnamic_weights=torch.ones(16).cuda()
        self.iter_nums=0
        self.adnamic_weight=adnamic_weight
    def icp_loss(self, teacher_output, student_output, targets):
        b, p, l = targets.shape
        targets = targets.view(-1, l)
        student_output_identity, student_output_classes, student_output_policy = student_output
        target_identity, target_classes = F.one_hot(targets[:, 0], student_output_identity.shape[1]).cuda(), F.one_hot(
            targets[:, 1], student_output_classes.shape[1]).cuda()
        target_identity, target_classes = target_identity.float(), target_classes.float()
        with torch.no_grad():
            teacher_output_identity, teacher_output_classes, teacher_output_policy = teacher_output
        """======================================================KD============================================"""
        kl_policy_loss = 0.
        ce_policy_loss = 0.
        indices = torch.arange(1,b*2,2)
        indices2= torch.arange(0,b*2,2)
        classes_nums=teacher_output_classes.shape[1]
        identity_nums=teacher_output_identity.shape[1]
        if self.adnamic_weight:
            policy_adnamic_weights = torch.stack([(targets[indices][:, i + 2] == teacher_output_policy[indices][:,
                                                                                 2 * i:2 * (i + 1)].argmax(1)).sum() /
                                                  indices.shape[0] for i in range(targets.shape[1] - 2)], 0)
            classes_adnamic_weights = torch.Tensor([(classes_nums * (
                        targets[:, 1] == teacher_output_classes.argmax(1)).sum() / targets.shape[0] - 1) / (
                                                                classes_nums - 1)]).cuda()
            identity_adnamic_weights = torch.Tensor([(identity_nums * (
                        targets[:, 0] == teacher_output_policy.argmax(1)).sum() / targets.shape[0] - 1) / (
                                                                 identity_nums - 1)]).cuda()
            adnamic_weights = torch.cat([identity_adnamic_weights, classes_adnamic_weights, policy_adnamic_weights], 0)
            with torch.no_grad():
                if self.iter_nums==0:
                    self.adnamic_weights=adnamic_weights
                else:
                    self.adnamic_weights=adnamic_weights*0.01+self.adnamic_weights*0.99
                    self.adnamic_weights[self.adnamic_weights<0]=0.
        for i in range(targets.shape[1] - 2):
            target_policy_one = F.one_hot(targets[indices][:, i + 2], 2).cuda().float()
            kl_policy_loss += (self.policy_kl_loss(student_output_policy[indices][:, 2 * i:2 * (i + 1)]-student_output_policy[indices2][:, 2 * i:2 * (i + 1)].detach(),
                                                  teacher_output_policy[indices][:, 2 * i:2 * (i + 1)]-teacher_output_policy[indices2][:, 2 * i:2 * (i + 1)].detach(),
                                                  target_policy_one)*self.adnamic_weights[i+2])
            ce_policy_loss += (self.policy_ce_loss(student_output_policy[indices][:, 2 * i:2 * (i + 1)]-student_output_policy[indices2][:, 2 * i:2 * (i + 1)].detach(),
                                                  target_policy_one, target_policy_one)*self.adnamic_weights[i+2])
        kl_identity_loss = self.identity_kl_loss(student_output_identity, teacher_output_identity, target_identity)*self.adnamic_weights[0]
        kl_classes_loss = self.classes_kl_loss(student_output_classes, teacher_output_classes, target_classes)*self.adnamic_weights[1]
        kl_loss = kl_identity_loss + kl_classes_loss + kl_policy_loss
        """======================================================CE============================================"""
        ce_identity_loss = self.identity_ce_loss(student_output_identity, target_identity, target_identity)
        ce_classes_loss = self.classes_ce_loss(student_output_classes, target_classes, target_classes)
        ce_loss = ce_identity_loss + ce_classes_loss + ce_policy_loss
        if self.iter_nums % 1000 == 0:
            print(kl_identity_loss.item(), kl_classes_loss.item(), kl_policy_loss.item(), ce_identity_loss.item(),
                  ce_classes_loss.item(), ce_policy_loss.item())
            print(self.adnamic_weights)
            self.iter_nums = 0
        self.iter_nums += 1
        total_loss = 0.
        for weight, loss in zip(self.kd_and_ce_weight, [kl_loss, ce_loss]):
            total_loss += (weight * loss)
        return total_loss

    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):
        student_linear_outputs = student_io_dict[self.student_linear_module_path][self.student_linear_module_io]
        teacher_linear_outputs = teacher_io_dict[self.teacher_linear_module_path][self.teacher_linear_module_io]
        b, p, l = targets.shape
        target = targets.view(-1, l)
        cls_target = target[:, 1]
        """======================================================CE Loss==================================================="""
        ce_loss = self.cross_entropy_loss(student_linear_outputs, cls_target.long())
        """======================================================KL Loss==================================================="""
        kl_loss = self.kldiv_loss(torch.log_softmax(student_linear_outputs / self.kl_temp, dim=1),
                                  torch.softmax(teacher_linear_outputs / self.kl_temp, dim=1))
        kl_loss *= (self.kl_temp ** 2)
        """===================================================Policy Loss==================================================="""
        student_policy_module_outputs = student_io_dict[self.student_policy_module_path][self.student_policy_module_io]
        teacher_policy_module_outputs = teacher_io_dict[self.teacher_policy_module_path][self.teacher_policy_module_io]
        policy_loss = self.icp_loss(teacher_policy_module_outputs, student_policy_module_outputs, targets)
        total_loss = 0.
        for loss_weight, loss in zip(self.loss_weights, [ce_loss, kl_loss, policy_loss]):
            total_loss += loss_weight * loss
        return total_loss
