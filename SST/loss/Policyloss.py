
import torch
from torch import nn
from torch.nn.functional import adaptive_avg_pool2d, adaptive_max_pool2d, normalize, cosine_similarity
from torchdistill.losses.single import register_single_loss

@register_single_loss
class AuxPolicyKDLoss(nn.CrossEntropyLoss):
    def __init__(self, module_path='policy_module', module_io='output', reduction='mean',feature_nums=128,policy_nums=7,**kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.module_path = module_path
        self.module_io = module_io
        self.linear=nn.Linear(feature_nums,2*policy_nums+2).cuda() # policy+classes+identity

    def forward(self, student_io_dict, teacher_io_dict, target,*args, **kwargs):
        policy_module_outputs = teacher_io_dict[self.module_path][self.module_io]
        b,c = policy_module_outputs.shape
        assert b%2==0,"the batchsize mod 2 should be zero!"
        b1=int(b/2)
        b2=int(b/2)
        b1_indices=torch.arange(b)%2==0
        b2_indices=torch.arange(b)%2!=0
        b1_output=policy_module_outputs[b1_indices] # original
        b2_output=policy_module_outputs[b2_indices] # augment
        b,p,policy_len=target.shape
        policy_len=policy_len-1
        target=target.view(-1,policy_len+1)
        b1_target=target[b1_indices]
        b2_target=target[b2_indices]
        b1_target=b1_target.unsqueeze(-1).expand(-1,-1,b1).transpose(0,2)
        b2_target=b2_target.unsqueeze(-1).expand(-1,-1,b2)
        target_matrix=torch.cat([b1_target,b2_target],1)
        b1_output = b1_output.unsqueeze(-1).expand(-1,-1,b1).transpose(0,2)
        b2_output = b2_output.unsqueeze(-1).expand(-1,-1,b2)
        output_matrix=torch.cat([b1_output,b2_output],1)
        learning_matrix=self.linear(output_matrix.transpose(1,2)).transpose(1,2)
        print(target_matrix.shape,learning_matrix.shape)
        return super().forward(cos_similarities, targets)


@register_single_loss
class SSKDLoss(nn.Module):
    """
    Loss of contrastive prediction as self-supervision task (auxiliary task)
    "Knowledge Distillation Meets Self-Supervision"
    Refactored https://github.com/xuguodong03/SSKD/blob/master/student.py
    """
    def __init__(self, student_linear_module_path, teacher_linear_module_path, student_ss_module_path,
                 teacher_ss_module_path, kl_temp, ss_temp, tf_temp, ss_ratio, tf_ratio,
                 student_linear_module_io='output', teacher_linear_module_io='output',
                 student_ss_module_io='output', teacher_ss_module_io='output',
                 loss_weights=None, reduction='batchmean', **kwargs):
        super().__init__()
        self.loss_weights = [1.0, 1.0, 1.0, 1.0] if loss_weights is None else loss_weights
        self.kl_temp = kl_temp
        self.ss_temp = ss_temp
        self.tf_temp = tf_temp
        self.ss_ratio = ss_ratio
        self.tf_ratio = tf_ratio
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction)
        self.kldiv_loss = nn.KLDivLoss(reduction=reduction)
        self.student_linear_module_path = student_linear_module_path
        self.student_linear_module_io = student_linear_module_io
        self.teacher_linear_module_path = teacher_linear_module_path
        self.teacher_linear_module_io = teacher_linear_module_io
        self.student_ss_module_path = student_ss_module_path
        self.student_ss_module_io = student_ss_module_io
        self.teacher_ss_module_path = teacher_ss_module_path
        self.teacher_ss_module_io = teacher_ss_module_io

    @staticmethod
    def compute_cosine_similarities(ss_module_outputs, normal_indices, aug_indices,
                                    three_forth_batch_size, one_forth_batch_size):
        normal_feat = ss_module_outputs[normal_indices]
        aug_feat = ss_module_outputs[aug_indices]
        normal_feat = normal_feat.unsqueeze(2).expand(-1, -1, three_forth_batch_size).transpose(0, 2)
        aug_feat = aug_feat.unsqueeze(2).expand(-1, -1, one_forth_batch_size)
        return cosine_similarity(aug_feat, normal_feat, dim=1)

    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):
        student_linear_outputs = student_io_dict[self.student_linear_module_path][self.student_linear_module_io]
        teacher_linear_outputs = teacher_io_dict[self.teacher_linear_module_path][self.teacher_linear_module_io]
        device = student_linear_outputs.device
        batch_size = student_linear_outputs.shape[0]
        three_forth_batch_size = int(batch_size * 3 / 4)
        one_forth_batch_size = batch_size - three_forth_batch_size
        normal_indices = (torch.arange(batch_size) % 4 == 0)
        aug_indices = (torch.arange(batch_size) % 4 != 0)
        ce_loss = self.cross_entropy_loss(student_linear_outputs[normal_indices], targets)
        kl_loss = self.kldiv_loss(torch.log_softmax(student_linear_outputs[normal_indices] / self.kl_temp, dim=1),
                                  torch.softmax(teacher_linear_outputs[normal_indices] / self.kl_temp, dim=1))
        kl_loss *= (self.kl_temp ** 2)

        # error level ranking
        aug_knowledges = torch.softmax(teacher_linear_outputs[aug_indices] / self.tf_temp, dim=1)
        aug_targets = targets.unsqueeze(1).expand(-1, 3).contiguous().view(-1)
        aug_targets = aug_targets[:three_forth_batch_size].long().to(device)
        ranks = torch.argsort(aug_knowledges, dim=1, descending=True)
        ranks = torch.argmax(torch.eq(ranks, aug_targets.unsqueeze(1)).long(), dim=1)  # groundtruth label's rank
        indices = torch.argsort(ranks)
        tmp = torch.nonzero(ranks, as_tuple=True)[0]
        wrong_num = tmp.numel()
        correct_num = three_forth_batch_size - wrong_num
        wrong_keep = int(wrong_num * self.tf_ratio)
        indices = indices[:correct_num + wrong_keep]
        distill_index_tf = torch.sort(indices)[0]

        student_ss_module_outputs = student_io_dict[self.student_ss_module_path][self.student_ss_module_io]
        teacher_ss_module_outputs = teacher_io_dict[self.teacher_ss_module_path][self.teacher_ss_module_io]

        s_cos_similarities = self.compute_cosine_similarities(student_ss_module_outputs, normal_indices,
                                                              aug_indices, three_forth_batch_size, one_forth_batch_size)
        t_cos_similarities = self.compute_cosine_similarities(teacher_ss_module_outputs, normal_indices,
                                                              aug_indices, three_forth_batch_size, one_forth_batch_size)
        t_cos_similarities = t_cos_similarities.detach()

        aug_targets = \
            torch.arange(one_forth_batch_size).unsqueeze(1).expand(-1, 3).contiguous().view(-1)
        aug_targets = aug_targets[:three_forth_batch_size].long().to(device)
        ranks = torch.argsort(t_cos_similarities, dim=1, descending=True)
        ranks = torch.argmax(torch.eq(ranks, aug_targets.unsqueeze(1)).long(), dim=1)  # groundtruth label's rank
        indices = torch.argsort(ranks)
        tmp = torch.nonzero(ranks, as_tuple=True)[0]
        wrong_num = tmp.numel()
        correct_num = three_forth_batch_size - wrong_num
        wrong_keep = int(wrong_num * self.ss_ratio)
        indices = indices[:correct_num+wrong_keep]
        distill_index_ss = torch.sort(indices)[0]

        ss_loss = self.kldiv_loss(torch.log_softmax(s_cos_similarities[distill_index_ss] / self.ss_temp, dim=1),
                                  torch.softmax(t_cos_similarities[distill_index_ss] / self.ss_temp, dim=1))
        ss_loss *= (self.ss_temp ** 2)
        log_aug_outputs = torch.log_softmax(student_linear_outputs[aug_indices] / self.tf_temp, dim=1)
        tf_loss = self.kldiv_loss(log_aug_outputs[distill_index_tf], aug_knowledges[distill_index_tf])
        tf_loss *= (self.tf_temp ** 2)
        total_loss = 0
        for loss_weight, loss in zip(self.loss_weights, [ce_loss, kl_loss, ss_loss, tf_loss]):
            total_loss += loss_weight * loss
        return total_loss
