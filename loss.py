# loss.py
import torch
import torch.nn.functional as F


def compute_kld_labeled(outputs_labeled, targets_labeled, num_classes=10, high=0.9):
    """
    KLD between model predictions and soft-targets for labeled samples
    """
    device = outputs_labeled.device
    batch_size = outputs_labeled.size(0)

    predictions = outputs_labeled.argmax(dim=1)
    log_probs = F.log_softmax(outputs_labeled, dim=1)

    soft_targets = torch.zeros(batch_size, num_classes, device=device, dtype=torch.float)

    for i in range(batch_size):
        if predictions[i] == targets_labeled[i]:
            base = (1.0 - high) / (num_classes - 1)
            soft_targets[i] = torch.full((num_classes,), base, device=device)
            soft_targets[i, targets_labeled[i]] = high
        else:
            soft_targets[i] = torch.full((num_classes,), 1.0 / num_classes, device=device)

    kld = F.kl_div(log_probs, soft_targets, reduction="batchmean")
    return kld


def compute_kld_unlabeled(
    unlabeled_logits,
    unlabeled_indices_batch,
    pseudo_storage,
    num_classes=10,
    high=0.9,
):
    """
    KLD for unlabeled samples with agreed pseudo-labels
    """
    if unlabeled_logits.numel() == 0:
        return torch.tensor(0.0, device=unlabeled_logits.device), 0

    device = unlabeled_logits.device
    batch_size = unlabeled_logits.size(0)

    predictions = unlabeled_logits.argmax(dim=1)
    log_probs = F.log_softmax(unlabeled_logits, dim=1)

    agreed_indices, agreed_labels = pseudo_storage.get_agreed_samples()
    agreed_dict = {
        int(idx): int(label)
        for idx, label in zip(agreed_indices, agreed_labels)
    }

    soft_targets = torch.zeros(batch_size, num_classes, device=device, dtype=torch.float)
    match_count = 0

    for i in range(batch_size):
        idx_val = int(
            unlabeled_indices_batch[i].item()
            if torch.is_tensor(unlabeled_indices_batch[i])
            else unlabeled_indices_batch[i]
        )

        if idx_val in agreed_dict:
            agreed_pseudo_label = agreed_dict[idx_val]
            if predictions[i].item() == agreed_pseudo_label:
                match_count += 1
                base = (1.0 - high) / (num_classes - 1)
                soft_targets[i] = torch.full((num_classes,), base, device=device)
                soft_targets[i, agreed_pseudo_label] = high
            else:
                soft_targets[i] = torch.full(
                    (num_classes,), 1.0 / num_classes, device=device
                )
        else:
            soft_targets[i] = torch.full(
                (num_classes,), 1.0 / num_classes, device=device
            )

    kld = F.kl_div(log_probs, soft_targets, reduction="batchmean")
    return kld, match_count
