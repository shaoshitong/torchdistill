import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.jit.annotations import Tuple, List

from torchdistill.common.constant import def_logger
from torchdistill.models.util import wrap_if_distributed, load_module_ckpt, save_module_ckpt, redesign_model
from torchdistill.models.special import register_special_module,SpecialModule
from torch.nn.functional import normalize
class Lambda(nn.Module):
    def __init__(self,lambda_function):
        super(Lambda, self).__init__()
        self.lambda_function=lambda_function
    def forward(self,x,*args,**kwargs):
        return self.lambda_function(x,*args,**kwargs)
@register_special_module
class WrapperPolicy(SpecialModule):
    def __init__(self, input_module, feat_dim,out_dim, policy_module_ckpt, device, device_ids, distributed, freezes_policy_module=False,
                 teacher_model=None, student_model=None,use_ckpt=False, **kwargs):
        super().__init__()
        is_teacher = teacher_model is not None
        if not is_teacher:
            student_model = wrap_if_distributed(student_model, device, device_ids, distributed)

        self.model = teacher_model if is_teacher else student_model
        self.is_teacher = is_teacher
        self.input_module_path = input_module['path']
        self.input_module_io = input_module['io']
        policy_module = nn.Sequential(
            nn.Linear(feat_dim, int((feat_dim+out_dim)//2)),
            nn.ReLU(inplace=True),
            nn.Linear(int((feat_dim+out_dim)//2), out_dim),
            Lambda(lambda x:normalize(x,dim=1))
        )
        self.ckpt_file_path = policy_module_ckpt
        if os.path.isfile(self.ckpt_file_path) and use_ckpt:
            s="teacher" if self.is_teacher else "student"
            print(f"successfully save the policy {s} module!")
            map_location = {'cuda:0': 'cuda:{}'.format(device_ids[0])} if distributed else device
            load_module_ckpt(policy_module, map_location, self.ckpt_file_path)
        self.use_ckpt=use_ckpt
        self.policy_module = policy_module if is_teacher and freezes_policy_module \
            else wrap_if_distributed(policy_module, device, device_ids, distributed)

    def forward(self, x):
        if self.is_teacher:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)

    def post_forward(self, io_dict):
        flat_outputs = torch.flatten(io_dict[self.input_module_path][self.input_module_io], 1)
        self.policy_module(flat_outputs)
    @torch.no_grad()
    def post_process(self, *args, **kwargs):
        if (not self.use_ckpt) and self.is_teacher and self.policy_module.requires_grad==True:
            s="teacher" if self.is_teacher else "student"
            print(f"successfully save the policy {s} module!")
            save_module_ckpt(self.policy_module, self.ckpt_file_path)

