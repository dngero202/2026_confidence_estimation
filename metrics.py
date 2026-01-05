#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics as sk_metrics


# In[ ]:


# =========================
# Evaluation Functions
# =========================
def calc_aurc_eaurc(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    sort_values = sorted(zip(softmax_max[:], correctness[:]), key=lambda x:x[0], reverse=True)
    sort_softmax_max, sort_correctness = zip(*sort_values)
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    aurc, eaurc = aurc_eaurc(risk_li)

    return aurc, eaurc

def calc_fpr_aupr(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    fpr, tpr, thresholds = sk_metrics.roc_curve(correctness, softmax_max)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_in_tpr_95 = fpr[idx_tpr_95]

    aupr_err = sk_metrics.average_precision_score(-1 * correctness + 1, -1 * softmax_max)

    print("AUPR {0:.2f}".format(aupr_err*100))
    print('FPR {0:.2f}'.format(fpr_in_tpr_95*100))

    return aupr_err, fpr_in_tpr_95

def calc_ece(softmax, label, bins=15):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmax = torch.tensor(softmax)
    labels = torch.tensor(label)

    softmax_max, predictions = torch.max(softmax, 1)
    correctness = predictions.eq(labels)

    ece = torch.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = softmax_max.gt(bin_lower.item()) * softmax_max.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            avg_confidence_in_bin = softmax_max[in_bin].mean()

            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    print("ECE {0:.2f} ".format(ece.item()*100))

    return ece.item()

def calc_nll_brier(softmax, logit, label, label_onehot):
    brier_score = np.mean(np.sum((softmax - label_onehot) ** 2, axis=1))

    logit = torch.tensor(logit, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.int)
    logsoftmax = torch.nn.LogSoftmax(dim=1)

    log_softmax = logsoftmax(logit)
    nll = calc_nll(log_softmax, label)

    print("NLL {0:.2f} ".format(nll.item()*10))
    print('Brier {0:.2f}'.format(brier_score*100))

    return nll.item(), brier_score

def calc_nll(log_softmax, label):
    out = torch.zeros_like(label, dtype=torch.float)
    for i in range(len(label)):
        out[i] = log_softmax[i][label[i]]

    return -out.sum()/len(out)

def coverage_risk(confidence, correctness):
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)
        coverage_list.append(coverage)

        if correctness[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))

    return risk_list, coverage_list

def aurc_eaurc(risk_list):
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)  # EXACT AS DOCUMENT 3
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area

    print("AURC {0:.2f}".format(aurc*1000))
    print("EAURC {0:.2f}".format(eaurc*1000))

    return aurc, eaurc


def calc_metrics(data_loader, model):
    model.eval()

    list_softmax = []
    list_correct = []
    list_logit = []
    label_list = []
    list_onehot = []
    
    with torch.no_grad():
        for inputs, targets, idx_list in data_loader:
            inputs, targets = inputs.cuda(), targets
            label_list.extend(targets)
            list_onehot.extend(F.one_hot(targets, num_classes=10).data.numpy())
            
            outputs, _ = model(inputs)  # Note: Your model returns (logits, embeddings)
            list_softmax.extend(F.softmax(outputs).cpu().data.numpy())  # EXACT AS DOCUMENT 3 - missing dim=1
            pred = outputs.data.max(1, keepdim=True)[1]
            
            for i in outputs:
                list_logit.append(i.cpu().data.numpy())
            
            for j in range(len(pred)):
                if pred[j] == targets[j]:
                    cor = 1
                else:
                    cor = 0
                list_correct.append(cor)
    
    list_onehot = np.array(list_onehot)
    
    aurc, eaurc = calc_aurc_eaurc(list_softmax, list_correct)
    aupr, fpr = calc_fpr_aupr(list_softmax, list_correct)
    ece = calc_ece(list_softmax, label_list, bins=15)
    nll, brier = calc_nll_brier(list_softmax, list_logit, label_list, list_onehot)
    
    return np.mean(list_correct), aurc, eaurc, aupr, fpr, ece, nll, brier

