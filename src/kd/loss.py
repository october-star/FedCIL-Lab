import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, old_classes, temperature=2.0):
    """
    L_distill = T^2 * KL(softmax(z_curr/T) || softmax(z_old/T)), only old classes
    """
    if old_classes <= 0:
        return student_logits.new_tensor(0.0)
    k = min(old_classes, teacher_logits.size(1), student_logits.size(1))
    if k <= 0:
        return student_logits.new_tensor(0.0)
    s = student_logits[:, :k] / temperature
    t = teacher_logits[:, :k] / temperature
    return (temperature ** 2) * F.kl_div(
        F.log_softmax(s, dim=1),
        F.softmax(t, dim=1),
        reduction="batchmean",
    )