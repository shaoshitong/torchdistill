from torchdistill.losses.single import register_single_loss,register_org_loss
import torch
import torch.nn as nn
import torch.nn.functional as F

@register_single_loss
class KFLoss(nn.Module):
    def __init__(self,factor_fkd=1.0,factor_fr=0.1):
        super(KFLoss, self).__init__()
        self.mm=lambda x:torch.matmul(x,x.T)
        self.adaptivepool2d=nn.AdaptiveAvgPool2d((1,1))
        self.flatten=nn.Flatten()
        self.factor_fkd=factor_fkd
        self.factor_fr=factor_fr
    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        student_list=[m['output'] for m in student_io_dict.values()]
        teacher_list=[m['output'] for m in teacher_io_dict.values()]
        assert len(student_list)==len(teacher_list),"the len should be same!"
        kfd_loss=0.
        for student_feature_map,teacher_feature_map in zip(student_list,teacher_list):
            sm=student_feature_map=self.flatten(self.adaptivepool2d(student_feature_map))
            norm_student=student_feature_map.pow(2).sum(1,keepdim=True).sqrt()
            student_feature_map=student_feature_map/norm_student
            s_k=self.mm(student_feature_map)

            with torch.no_grad():
                teacher_feature_map=self.flatten(self.adaptivepool2d(teacher_feature_map))
                norm_teacher=teacher_feature_map.pow(2).sum(1,keepdim=True).sqrt()
                teacher_feature_map=teacher_feature_map/norm_teacher
                t_k=self.mm(teacher_feature_map)
            distill_loss=(s_k-t_k).pow(2).mean()
            loss=distill_loss*self.factor_fkd+sm.pow(2).mean()*self.factor_fr
            kfd_loss+=loss
        return kfd_loss








