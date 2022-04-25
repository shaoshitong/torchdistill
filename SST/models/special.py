import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.jit.annotations import Tuple, List
import einops
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
                 teacher_model=None, student_model=None,use_ckpt=False, identity=False,
                 **kwargs):
        super().__init__()
        is_teacher = teacher_model is not None
        if not is_teacher:
            student_model = wrap_if_distributed(student_model, device, device_ids, distributed)

        self.model = teacher_model if is_teacher else student_model
        self.is_teacher = is_teacher
        self.input_module_path = input_module['path']
        self.input_module_io = input_module['io']
        if identity:
            policy_module=nn.Identity()
        else:
            policy_module = nn.Sequential(
                nn.Linear(feat_dim, int((feat_dim+out_dim)//2)),
                nn.ReLU(inplace=True),
                nn.Linear(int((feat_dim+out_dim)//2), out_dim),
                Lambda(lambda x:normalize(x,dim=1))
            )
        self.ckpt_file_path = policy_module_ckpt
        if os.path.isfile(self.ckpt_file_path) and use_ckpt and identity==False:
            s="teacher" if self.is_teacher else "student"
            print(f"successfully load the policy {s} module!")
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
        if isinstance(self.policy_module,nn.Identity):
            return None
        if (not self.use_ckpt) and self.is_teacher and self.policy_module._modules[list(self.policy_module._modules.keys())[0]].weight.requires_grad:
            s="teacher" if self.is_teacher else "student"
            print(f"successfully save the policy {s} module!")
            save_module_ckpt(self.policy_module, self.ckpt_file_path)


class ICPmodule(nn.Module):
    def __init__(self,feat_dim,identity_nums,classes_nums,policy_nums):
        super(ICPmodule,self).__init__()
        self.feat_dim=feat_dim
        self.conv=nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, (3, 3), (1, 1), (1, 1), bias=False, groups=feat_dim),
            nn.Conv2d(feat_dim, 3 * feat_dim, (1, 1), (1, 1), (0, 0), bias=False),
        )
        self.normalize=nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            Lambda(lambda x: normalize(x, dim=1))
        )
        self.linear1=nn.Linear(feat_dim,identity_nums)
        self.linear2=nn.Linear(feat_dim,classes_nums)
        self.linear3=nn.Linear(feat_dim,policy_nums)

    def forward(self,x):
        x=self.conv(x)+einops.rearrange(x.unsqueeze(1).expand(-1,3,-1,-1,-1),"a b c d e -> a (b c) d e")
        x=self.normalize(x)
        identity,classes,policy=torch.chunk(x,dim=1,chunks=3)
        return self.linear1(identity),self.linear2(classes),self.linear3(policy)


@register_special_module
class WrapperICP(SpecialModule):
    def __init__(self, input_module, feat_dim, policy_module_ckpt, device, device_ids, distributed,
                 identity_num=50000,classes_num=10,policy_num=16384,
                 freezes_policy_module=False,teacher_model=None, student_model=None,use_ckpt=False, identity=False,
                 **kwargs):
        super().__init__()
        is_teacher = teacher_model is not None
        if not is_teacher:
            student_model = wrap_if_distributed(student_model, device, device_ids, distributed)

        self.model = teacher_model if is_teacher else student_model
        self.is_teacher = is_teacher
        self.input_module_path = input_module['path']
        self.input_module_io = input_module['io']
        if identity:
            policy_module=nn.Identity()
        else:
            policy_module = ICPmodule(feat_dim,identity_nums=identity_num,classes_nums=classes_num,policy_nums=policy_num)
        self.ckpt_file_path = policy_module_ckpt
        if os.path.isfile(self.ckpt_file_path) and use_ckpt and identity==False:
            s="teacher" if self.is_teacher else "student"
            print(f"successfully load the policy {s} module!")
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
        flat_outputs = io_dict[self.input_module_path][self.input_module_io]
        self.policy_module(flat_outputs)
    @torch.no_grad()
    def post_process(self, *args, **kwargs):
        if isinstance(self.policy_module,nn.Identity):
            return None
        if (not self.use_ckpt) and self.is_teacher and self.policy_module.conv._modules[list(self.policy_module.conv._modules.keys())[0]].weight.requires_grad:
            s="teacher" if self.is_teacher else "student"
            print(f"successfully save the policy {s} module!")
            save_module_ckpt(self.policy_module, self.ckpt_file_path)

