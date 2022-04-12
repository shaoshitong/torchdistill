
from torchdistill.losses.util import register_func2extract_org_output
@register_func2extract_org_output
def extract_simple_org_loss_with_supp_dict(org_criterion, student_outputs, teacher_outputs, targets, uses_teacher_output,supp_dict,**kwargs):
    org_loss_dict = dict()
    if org_criterion is not None:
        # Models with auxiliary classifier returns multiple outputs
        if isinstance(student_outputs, (list, tuple)):
            if uses_teacher_output:
                for i, sub_student_outputs, sub_teacher_outputs in enumerate(zip(student_outputs, teacher_outputs)):
                    org_loss_dict[i] = org_criterion(sub_student_outputs, sub_teacher_outputs, targets,supp_dict)
            else:
                for i, sub_outputs in enumerate(student_outputs):
                    org_loss_dict[i] = org_criterion(sub_outputs, targets,supp_dict)
        else:
            org_loss = org_criterion(student_outputs, teacher_outputs, targets,supp_dict) if uses_teacher_output \
                else org_criterion(student_outputs, targets,supp_dict)
            org_loss_dict = {0: org_loss}
    return org_loss_dict


