import torch
from torchdistill.core.forward_proc import register_forward_proc_func

@register_forward_proc_func
def forward_batch_target_policy(model, sample_batch, targets, supp_dict=None):
    c,h,w=sample_batch.shape[:-3]
    sample_batch=sample_batch.view(-1,c,h,w)
    assert targets.ndim==1,"the ndim should be 1"
    targets=targets.unsqueeze(-1).expand(1,2).view(-1)
    # supp_dict['policy_index']=
    return model(sample_batch, targets)