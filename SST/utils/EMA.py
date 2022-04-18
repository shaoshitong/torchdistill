import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_optimizer import AdaBound
class EMA(object):
    def __init__(self,teacher_model_list,student_model_list,momentum=0.9):
        super(EMA, self).__init__()
        self.teacher_model_list=teacher_model_list
        self.student_model_list=student_model_list
        self.momentum=momentum

    @torch.no_grad()
    def step(self):
        for teacher_model,student_model in zip(self.teacher_model_list,self.student_model_list):
            for teacher_parameter,student_parameter in zip(teacher_model.parameters(),student_model.parameters()):
                    teacher_parameter.mul_(self.momentum).add_((1.0 - self.momentum)*student_parameter)
