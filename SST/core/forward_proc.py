import torch
from torchdistill.core.forward_proc import register_forward_proc_func

@register_forward_proc_func
def forward_batch_target_policy(model, sample_batch, targets, supp_dict=None):
    c,h,w=sample_batch.shape[-3:]
    sample_batch=sample_batch.view(-1,c,h,w)
    return model(sample_batch)
