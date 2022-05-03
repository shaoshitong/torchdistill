from torchdistill.models.util import wrap_if_distributed, load_module_ckpt, save_module_ckpt, redesign_model
import torch
from torch import nn
from torch.nn.functional import adaptive_avg_pool2d, adaptive_max_pool2d, normalize, cosine_similarity
from torchdistill.losses.single import register_single_loss
import einops
@register_single_loss
class AuxPolicyKDLoss(nn.Module):
    def __init__(self, module_path='policy_module', module_io='output', reduction='mean',feature_nums=128,policy_nums=7,
                 type="mse",ckpt_file_path="policy_linear.pth",negative_loss_weight=[1.0,0.5,0.001],positive_loss_weight=[1.0,0.5,0.001],
                 num_classes=10,
                 p_t=5,**kwargs):
        super().__init__()
        self.module_path = module_path
        self.module_io = module_io
        if type=="mse":
            self.linear=nn.Linear(feature_nums,policy_nums+2).cuda() # policy+classes+identity
            self.mse=nn.MSELoss(reduction='none')
            self.kl=nn.KLDivLoss(reduction="mean")
            self.bce=nn.BCEWithLogitsLoss(reduction="mean")
        else:
            self.linear=nn.Linear(feature_nums,2*(policy_nums+2)).cuda() # policy+classes+identity
        self.type=type
        # nn.init.zeros_(self.linear.bias)
        # nn.init.trunc_normal_(self.linear.weight,0,0.001)
        self.positive_loss_weight=positive_loss_weight
        self.negative_loss_weight=negative_loss_weight
        self.p_t=p_t
        self.ckpt_file_path=ckpt_file_path
        self.num_classes=num_classes
        self.iter=0
        print(f"p loss weight is {positive_loss_weight},{negative_loss_weight}")
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
        target=target.view(-1,policy_len)
        b1_target=target[b1_indices]
        b2_target=target[b2_indices]
        identiy_value=1/(b1-1)+1
        classes_value=1/(self.num_classes-1)+1
        b1_target=b1_target.unsqueeze(-1).expand(-1,-1,b1).transpose(0,2)
        b2_target=b2_target.unsqueeze(-1).expand(-1,-1,b2)
        target_matrix=(b1_target==b2_target).float()
        target_matrix_classes=target_matrix[:,0,:]*classes_value-1/(self.num_classes-1)
        target_matrix_policy=einops.rearrange(target_matrix[:,1:,:],"a b c -> (a c) b")*2-1
        target_matrix_identity=torch.eye(target_matrix.shape[0]).to(target_matrix.device).float()*identiy_value-1/(b1-1)
        if self.type=="mse":
            b1_output = b1_output.unsqueeze(-1).expand(-1, -1, b1).transpose(0, 2)
            b2_output = b2_output.unsqueeze(-1).expand(-1, -1, b2)
            output_matrix = torch.cat([b1_output, b2_output], 1)
            learning_matrix = self.linear(output_matrix.transpose(1, 2)).transpose(1, 2)
            learning_matrix_identity=learning_matrix[:,0,:]
            learning_matrix_classes=learning_matrix[:,1,:]
            learning_matrix_policy=einops.rearrange(learning_matrix[:,2:,:],"a b c -> (a c) b")
            identity_loss=self.mse(learning_matrix_identity,target_matrix_identity)
            classes_loss=self.mse(learning_matrix_classes,target_matrix_classes)
            policy_loss = self.mse(learning_matrix_policy,target_matrix_policy)
        else:
            raise NotImplementedError
        self.save()
        identity_loss = torch.where(target_matrix_identity == 1., identity_loss * self.positive_loss_weight[0],
                                    identity_loss * self.negative_loss_weight[0]).mean()
        classes_loss = torch.where(target_matrix_classes == 1., classes_loss * self.positive_loss_weight[1],
                                    classes_loss * self.negative_loss_weight[1]).mean()
        policy_loss = torch.where(target_matrix_policy == 1., policy_loss * self.positive_loss_weight[2],
                                    policy_loss * self.negative_loss_weight[2]).mean()
        return identity_loss+classes_loss+policy_loss
    @torch.no_grad()
    def save(self, *args, **kwargs):
        # print("successfully save the linear policy module")
        save_module_ckpt(self.linear, self.ckpt_file_path)

@register_single_loss
class PolicyLoss(nn.Module):
    def __init__(self, student_linear_module_path, teacher_linear_module_path, student_policy_module_path,
                 teacher_policy_module_path, kl_temp, policy_temp,policy_ratio,ckpt_file_path,
                 feature_nums=128, policy_nums=7,num_classes=10,option=0,
                 student_linear_module_io='output', teacher_linear_module_io='output',
                 student_policy_module_io='output', teacher_policy_module_io='output',
                 loss_weights=None, reduction='mean',type='mse',freeze_student=False,
                 p_t=5,positive_loss_weight=[1.0,0.5,0.001],negative_loss_weight=[1.0,0.5,0.001],
                 kd_and_ce_weight=[1,1],**kwargs):
        super().__init__()
        self.loss_weights = [1.0, 1.0, 1.0] if loss_weights is None else loss_weights
        print("ce,kd,policy loss weight is",self.loss_weights)
        print("positive and negative loss weight is",positive_loss_weight,negative_loss_weight)
        print("policy's kd and ce weight is ",kd_and_ce_weight)
        self.kl_temp = kl_temp
        self.policy_temp = policy_temp
        self.policy_ratio = policy_ratio
        self.p_t=p_t
        self.option=option
        self.num_classes=num_classes
        self.positive_loss_weight=positive_loss_weight
        self.negative_loss_weight=negative_loss_weight
        self.kd_and_ce_weight=kd_and_ce_weight
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction)
        self.kldiv_loss = nn.KLDivLoss(reduction=reduction)
        self.student_linear_module_path = student_linear_module_path
        self.student_linear_module_io = student_linear_module_io
        self.teacher_linear_module_path = teacher_linear_module_path
        self.teacher_linear_module_io = teacher_linear_module_io
        self.student_policy_module_path = student_policy_module_path
        self.student_policy_module_io = student_policy_module_io
        self.teacher_policy_module_path = teacher_policy_module_path
        self.teacher_policy_module_io = teacher_policy_module_io
        if type=="mse":
            self.linear1=nn.Linear(feature_nums,policy_nums+2).cuda() # policy+classes+identity
            self.linear2=nn.Linear(feature_nums,policy_nums+2).cuda() # policy+classes+identity
            self.mse=nn.MSELoss(reduction='none')
            self.bce=nn.BCEWithLogitsLoss(reduction="mean")
            self.kd=nn.KLDivLoss(reduction="mean")
        else:
            raise NotImplementedError
        self.type=type
        self.ckpt_file_path=ckpt_file_path
        map_location = {'cuda:0': 'cuda:0'}
        print("successfully load the teacher linear policy module")
        load_module_ckpt(self.linear1,map_location,self.ckpt_file_path)
        self.linear1.weight.requires_grad=False
        self.linear1.bias.requires_grad=False
        if freeze_student:
            load_module_ckpt(self.linear2,map_location,self.ckpt_file_path)
            print("successfully load the student linear policy module")
            # nn.init.trunc_normal_(self.linear2.weight, 0, 0.001)
            # nn.init.zeros_(self.linear2.bias)
            self.linear2.weight.requires_grad=False
            self.linear2.bias.requires_grad=False
        else:
            # nn.init.trunc_normal_(self.linear2.weight, 0, 0.001)
            # nn.init.zeros_(self.linear2.bias)
            pass
    def policy_loss(self,teacher_output,student_output,targets,b1_indices,b2_indices,b1,b2):
        student_b1_output=student_output[b1_indices]
        student_b2_output=student_output[b2_indices]
        student_b1_output = student_b1_output.unsqueeze(-1).expand(-1, -1, b1).transpose(0, 2)
        student_b2_output = student_b2_output.unsqueeze(-1).expand(-1, -1, b2)
        student_output_matrix = torch.cat([student_b1_output, student_b2_output], 1)
        student_learning_matrix = self.linear2(student_output_matrix.transpose(1, 2)).transpose(1, 2)
        student_matrix_identity = student_learning_matrix[:, 0, :]
        student_matrix_classes =  student_learning_matrix[:, 1, :]
        student_matrix_policy = einops.rearrange(student_learning_matrix[:, 2:, :], "a b c -> (a c) b")
        b,p,policy_len=targets.shape
        identiy_value=1/(b1-1)+1
        classes_value=1/(self.num_classes-1)+1
        policy_len=policy_len-1
        targets=targets.view(-1,policy_len+1)
        b1_target=targets[b1_indices]
        b2_target=targets[b2_indices]
        b1_target=b1_target.unsqueeze(-1).expand(-1,-1,b1).transpose(0,2)
        b2_target=b2_target.unsqueeze(-1).expand(-1,-1,b2)
        target_matrix=(b1_target==b2_target).float()
        target_matrix_classes=target_matrix[:,0,:]*classes_value-1/(self.num_classes-1)
        target_matrix_policy=einops.rearrange(target_matrix[:,1:,:],"a b c -> (a c) b")*2-1
        target_matrix_identity=torch.eye(target_matrix.shape[0]).to(target_matrix.device).float()*identiy_value-1/(b1-1)
        with torch.no_grad():
            teacher_b1_output = teacher_output[b1_indices]
            teacher_b2_output = teacher_output[b2_indices]
            teacher_b1_output = teacher_b1_output.unsqueeze(-1).expand(-1, -1, b1).transpose(0, 2)
            teacher_b2_output = teacher_b2_output.unsqueeze(-1).expand(-1, -1, b2)
            teacher_output_matrix = torch.cat([teacher_b1_output, teacher_b2_output], 1)
            teacher_learning_matrix = self.linear1(teacher_output_matrix.transpose(1, 2)).transpose(1, 2)
            teacher_matrix_identity = teacher_learning_matrix[:, 0, :]
            teacher_matrix_classes = teacher_learning_matrix[:, 1, :]
            teacher_matrix_policy = einops.rearrange(teacher_learning_matrix[:, 2:, :], "a b c -> (a c) b")
        teacher_matrix_identity = torch.where(torch.eye(b1).cuda().float() == 0., teacher_matrix_identity.detach(),
                                              teacher_matrix_identity)
        student_matrix_identity = torch.where(torch.eye(b1).cuda().float() == 0., student_matrix_identity.detach(),
                                              student_matrix_identity)
        if self.type=="mse":
            kl_identity_loss=self.mse(student_matrix_identity,teacher_matrix_identity)
            kl_classes_loss=self.mse(student_matrix_classes,teacher_matrix_classes)
            kl_policy_loss=self.mse(student_matrix_policy,teacher_matrix_policy)
            kl_identity_loss = torch.where(target_matrix_identity == 1., kl_identity_loss * self.positive_loss_weight[0],
                                        kl_identity_loss * self.negative_loss_weight[0]).mean()
            kl_classes_loss = torch.where(target_matrix_classes == 1., kl_classes_loss * self.positive_loss_weight[1],
                                       kl_classes_loss * self.negative_loss_weight[1]).mean()
            kl_policy_loss = torch.where(target_matrix_policy == 1., kl_policy_loss * self.positive_loss_weight[2],
                                      kl_policy_loss * self.negative_loss_weight[2]).mean()
            kl_loss=kl_identity_loss + kl_classes_loss + kl_policy_loss
        else:
            raise NotImplementedError
        if self.type=="mse":
            ce_identity_loss=self.kd(student_matrix_identity,target_matrix_identity)
            ce_classes_loss=self.kd(student_matrix_classes,target_matrix_classes)
            ce_policy_loss = self.mse(student_matrix_policy,target_matrix_policy)
            ce_identity_loss = torch.where(target_matrix_identity == 1., ce_identity_loss * self.positive_loss_weight[0],
                                        ce_identity_loss * self.negative_loss_weight[0]).mean()
            ce_classes_loss = torch.where(target_matrix_classes == 1., ce_classes_loss * self.positive_loss_weight[1],
                                       ce_classes_loss * self.negative_loss_weight[1]).mean()
            ce_policy_loss = torch.where(target_matrix_policy == 1., ce_policy_loss * self.positive_loss_weight[2],
                                      ce_policy_loss * self.negative_loss_weight[2]).mean()
            ce_loss=ce_identity_loss + ce_classes_loss + ce_policy_loss
        else:
            raise NotImplementedError
        total_loss=0.
        for weight,loss in zip(self.kd_and_ce_weight,[kl_loss,ce_loss]):
            total_loss+=(weight*loss)
        return total_loss
    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):
        student_linear_outputs = student_io_dict[self.student_linear_module_path][self.student_linear_module_io]
        teacher_linear_outputs = teacher_io_dict[self.teacher_linear_module_path][self.teacher_linear_module_io]
        b, c = student_linear_outputs.shape
        assert b % 2 == 0, "the batchsize mod 2 should be zero!"
        b1 = int(b / 2)
        b2 = int(b / 2)
        b1_indices = torch.arange(b) % 2 == 0
        b2_indices = torch.arange(b) % 2 != 0
        b,p,policy_len=targets.shape
        policy_len=policy_len-1
        target=targets.view(-1,policy_len+1)
        cls_target=target[:,0]
        """======================================================CE Loss==================================================="""
        ce_loss_origin = self.cross_entropy_loss(student_linear_outputs[b1_indices], cls_target[b1_indices].long())
        ce_loss_augment = self.cross_entropy_loss(student_linear_outputs[b2_indices], cls_target[b2_indices].long())
        if self.option==0:
            ce_loss=(ce_loss_origin+ce_loss_augment)/2
        elif self.option==1:
            ce_loss=ce_loss_origin
            return ce_loss
        else:
            ce_loss=ce_loss_origin
        """======================================================KL Loss==================================================="""
        kl_loss_origin = self.kldiv_loss(torch.log_softmax(student_linear_outputs[b1_indices] / self.kl_temp, dim=1),
                                  torch.softmax(teacher_linear_outputs[b1_indices] / self.kl_temp, dim=1))
        kl_loss_origin *= (self.kl_temp ** 2)
        kl_loss_augment = self.kldiv_loss(torch.log_softmax(student_linear_outputs[b2_indices] / self.kl_temp, dim=1),
                                  torch.softmax(teacher_linear_outputs[b2_indices] / self.kl_temp, dim=1))
        kl_loss_augment *= (self.kl_temp ** 2)
        if self.option==2:
            kl_loss=kl_loss_origin
            return ce_loss+kl_loss
        elif self.option==3:
            kl_loss=kl_loss_augment
            return ce_loss+kl_loss
        else:
            kl_loss=(kl_loss_augment+kl_loss_origin)/2

        """===================================================Policy Loss==================================================="""
        student_policy_module_outputs = student_io_dict[self.student_policy_module_path][self.student_policy_module_io]
        teacher_policy_module_outputs = teacher_io_dict[self.teacher_policy_module_path][self.teacher_policy_module_io]
        policy_loss = self.policy_loss(teacher_policy_module_outputs,student_policy_module_outputs,targets,b1_indices,b2_indices,b1,b2)
        total_loss=0.
        for loss_weight, loss in zip(self.loss_weights, [ce_loss, kl_loss,policy_loss]):
            total_loss += loss_weight * loss
        return total_loss
