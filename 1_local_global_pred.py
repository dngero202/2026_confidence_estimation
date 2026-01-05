from model import resnet

import torchvision.transforms as transforms
import utils
from PIL import Image
from torch.utils.data import Dataset
import argparse
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from collections import Counter
import time
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import random

import metrics

from loader import (
    cifar10_dataset_labeled,
    cifar10_dataset_unlabeled,
)

from loss import compute_kld_labeled, compute_kld_unlabeled
from metrics import calc_metrics

import yaml
from types import SimpleNamespace

# =========================
# Seed Setting Function
# =========================
def set_seed(seed=42):
    """Set seed for reproducibility across all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Random seed set to {seed} for reproducibility\n")

# Load YAML config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Convert dict to object-style access (cfg.save_folder)
args = SimpleNamespace(**cfg)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    ),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    ),
])

trainset = cifar10_dataset_labeled(
    img_file=f'./cifar10_data/img_labeled_{args.label_size}.npy',
    label_file=f'./cifar10_data/ann_labeled_{args.label_size}.npy',
    label_size=args.label_size,
    train=True,
    train_transforms=transform_train,
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True,
    num_workers=10,
    pin_memory=True,
)

testset = cifar10_dataset_labeled(
    img_file='./cifar10_data/img_test.npy',
    label_file='./cifar10_data/ann_test.npy',
    label_size=10000,
    train=False,
    test_transforms=transform_test,
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=128,
    shuffle=False,
    num_workers=10,
    pin_memory=True,
)

unlabeled_trainset = cifar10_dataset_unlabeled(
    './cifar10_data/img_unlabeled_' + str(args.label_size) + '.npy',
    './cifar10_data/ann_unlabeled_' + str(args.label_size) + '.npy',
    label_size=args.label_size,
    train_transforms=transform_train,
    limit=args.unlabeled_limit
)

unlabeled_loader = torch.utils.data.DataLoader(
    unlabeled_trainset,
    batch_size=128,
    shuffle=True,
    num_workers=10,
    pin_memory=True,
)

# =========================
# ResNet Wrapper
# =========================
class ResNetWrapper(nn.Module):
    """Simple wrapper to handle return_embedding flag"""

    def __init__(self, base_model):
        super(ResNetWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, x, return_embedding=False):
        logits, embeddings = self.base_model(x)
        if return_embedding:
            return embeddings
        return logits, embeddings
    
# =========================
# Enhanced Embedding Accumulator
# =========================
class EmbeddingAccumulator:
    """Accumulate embeddings, logits, indices, and labels across batches"""

    def __init__(self):
        self.correct_embeddings = []
        self.correct_labels = []
        self.correct_indices = []
        self.unlabeled_embeddings = []
        self.unlabeled_logits = []
        self.unlabeled_indices = []

    def add_labeled(self, embeddings, logits, predictions, targets, indices):
        """Add correctly classified labeled embeddings (A1)"""
        embeddings = embeddings.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        indices = indices.cpu().numpy() if torch.is_tensor(indices) else np.array(indices)

        for i in range(len(targets)):
            if predictions[i] == targets[i]:
                self.correct_embeddings.append(embeddings[i])
                self.correct_labels.append(targets[i])
                self.correct_indices.append(indices[i])

    def add_unlabeled(self, embeddings, logits, indices):
        """Add all unlabeled embeddings (A2)"""
        embeddings = embeddings.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        indices = indices.cpu().numpy() if torch.is_tensor(indices) else np.array(indices)

        for i in range(len(embeddings)):
            self.unlabeled_embeddings.append(embeddings[i])
            self.unlabeled_logits.append(logits[i])
            self.unlabeled_indices.append(indices[i])

    def get_accumulated_data(self):
        """Get all accumulated data as numpy arrays"""
        return {
            'correct_embeddings': np.array(self.correct_embeddings) if self.correct_embeddings else np.array([]),
            'correct_labels': np.array(self.correct_labels) if self.correct_labels else np.array([]),
            'correct_indices': np.array(self.correct_indices) if self.correct_indices else np.array([]),
            'unlabeled_embeddings': np.array(self.unlabeled_embeddings) if self.unlabeled_embeddings else np.array([]),
            'unlabeled_logits': np.array(self.unlabeled_logits) if self.unlabeled_logits else np.array([]),
            'unlabeled_indices': np.array(self.unlabeled_indices) if self.unlabeled_indices else np.array([])
        }

    def reset(self):
        """Reset all accumulated data"""
        self.correct_embeddings = []
        self.correct_labels = []
        self.correct_indices = []
        self.unlabeled_embeddings = []
        self.unlabeled_logits = []
        self.unlabeled_indices = []

def generate_global_pseudo_labels_single_prototype(
        accumulator, num_classes=10):
    """Generate global pseudo-labels using MEAN prototype per class.
       Pseudo-label = class ID with maximum cosine similarity.
    """
    data = accumulator.get_accumulated_data()

    correct_embeddings = data['correct_embeddings']
    correct_labels = data['correct_labels']
    unlabeled_embeddings = data['unlabeled_embeddings']
    
    if len(correct_embeddings) == 0 or len(unlabeled_embeddings) == 0:
        return np.array([]), np.array([])

    prototypes = []
    prototype_labels = []

    # ----- Build mean prototypes -----
    for class_id in range(num_classes):
        class_mask = (correct_labels == class_id)
        class_embeddings = correct_embeddings[class_mask]

        if len(class_embeddings) == 0:
            prototype = np.zeros(correct_embeddings.shape[1])
        else:
            prototype = np.mean(class_embeddings, axis=0)

        prototypes.append(prototype)
        prototype_labels.append(class_id)

    prototypes = np.array(prototypes)
    prototype_labels = np.array(prototype_labels)

    # ----- Assign pseudo-labels -----
    global_pseudo_labels = []

    for emb in unlabeled_embeddings:
        similarities = np.dot(prototypes, emb) / (
            np.linalg.norm(prototypes, axis=1) * np.linalg.norm(emb) + 1e-8
        )

        best_idx = np.argmax(similarities)
        pseudo_label = prototype_labels[best_idx]

        global_pseudo_labels.append(pseudo_label)

    return np.array(global_pseudo_labels)

# =========================
# Local Pseudo-Labeling
# =========================
def generate_local_pseudo_labels_propagation(accumulator, k_neighbors=70, alpha=0.5,
                                             num_classes=10, max_iter=50):
    """Generate local pseudo-labels using label propagation"""
    data = accumulator.get_accumulated_data()

    correct_embeddings = data['correct_embeddings']
    correct_labels = data['correct_labels']
    unlabeled_embeddings = data['unlabeled_embeddings']

    if len(correct_embeddings) == 0 or len(unlabeled_embeddings) == 0:
        return np.array([])

    all_embeddings = np.vstack([correct_embeddings, unlabeled_embeddings])
    n_labeled = len(correct_embeddings)
    n_total = len(all_embeddings)

    sigma_squared = 1.25
    
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, n_total), 
                           algorithm='auto').fit(all_embeddings)
    distances, indices = nbrs.kneighbors(all_embeddings)

    W = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j_idx, j in enumerate(indices[i][1:]):
            dist = distances[i][j_idx + 1]
            W[i, j] = np.exp(-dist**2 / (2 * sigma_squared))
            W[j, i] = W[i, j]

    d = np.sum(W, axis=1)
    
    d_inv_sqrt = np.zeros(n_total)
    for i in range(n_total):
        if d[i] > 1e-10:
            d_inv_sqrt[i] = 1.0 / np.sqrt(d[i])
        else:
            d_inv_sqrt[i] = 0.0
    
    D_inv_sqrt = np.diag(d_inv_sqrt)
    S = D_inv_sqrt @ W @ D_inv_sqrt

    Y = np.zeros((n_total, num_classes))
    for i in range(n_labeled):
        Y[i, correct_labels[i]] = 1.0
    
    Y_init = Y.copy()
    F = Y.copy()
    
    for iteration in range(max_iter):
        F_new = alpha * (S @ F) + (1 - alpha) * Y_init
        
        diff = np.linalg.norm(F_new - F, 'fro')
        F = F_new
        
        if diff < 1e-6:
            break

    local_pseudo_labels = np.argmax(F[n_labeled:], axis=1)
    
    return local_pseudo_labels

# =========================
# Pseudo-Label Storage
# =========================
class PseudoLabelStorage:
    """Store pseudo-labels indexed by sample ID"""

    def __init__(self):
        self.global_pseudo = {}
        self.local_pseudo = {}

    def update(self, unlabeled_indices, global_labels, local_labels):
        """Store pseudo-labels by index"""
        for i, idx in enumerate(unlabeled_indices):
            self.global_pseudo[int(idx)] = int(global_labels[i])
            self.local_pseudo[int(idx)] = int(local_labels[i])

    def get_agreed_samples(self):
        """Return indices whose global and local pseudo-labels match"""
        agreed_indices = []
        agreed_labels = []

        for idx in self.global_pseudo.keys():
            if idx in self.local_pseudo:
                global_label = self.global_pseudo[idx]
                local_label = self.local_pseudo[idx]

                if global_label == local_label:
                    agreed_indices.append(idx)
                    agreed_labels.append(global_label)

        return np.array(agreed_indices), np.array(agreed_labels)

    def reset(self):
        """Reset all stored pseudo-labels"""
        self.global_pseudo = {}
        self.local_pseudo = {}

# =========================
# PHASE 1
# =========================
def phase1_supervised_training(labeledloader, model, optimizer, epoch):
    """Phase 1: Train on LABELED DATA ONLY with CE + KLD_labeled"""
    print(f"\n{'='*70}")
    print(f"PHASE 1: SUPERVISED TRAINING - Labeled Data Only")
    print(f"{'='*70}\n")
    
    model.train()
    
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    ce_losses = utils.AverageMeter()
    kld_labeled_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    end = time.time()
    
    for i, (input, target, idx) in enumerate(labeledloader):
        data_time.update(time.time() - end)
        
        input, target = input.cuda(), target.cuda()
        
        output, embeddings = model(input)
        
        ce_loss = F.cross_entropy(output, target)
        kld_lab = compute_kld_labeled(output, target, num_classes=10, high=args.confidence_high)
        loss = ce_loss + kld_lab
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        prec, correct = utils.accuracy(output, target)
        total_losses.update(loss.item(), input.size(0))
        ce_losses.update(ce_loss.item(), input.size(0))
        kld_labeled_losses.update(kld_lab.item(), input.size(0))
        top1.update(prec.item(), input.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}] PHASE1\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'CE {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                  'KLD_L {kld_lab.val:.4f} ({kld_lab.avg:.4f})\t'
                  'Prec {top1.val:.2f}% ({top1.avg:.2f}%)'.format(
                   epoch, i, len(labeledloader),
                   batch_time=batch_time, loss=total_losses, ce_loss=ce_losses,
                   kld_lab=kld_labeled_losses, top1=top1))
    
    return total_losses.avg, ce_losses.avg, kld_labeled_losses.avg, top1.avg

# =========================
# PHASE 2
# =========================
def phase2_accumulation_and_pseudo_labels(labeledloader, unlabeledloader, model, 
                                          accumulator, pseudo_storage, epoch):
    """Phase 2: EVAL MODE - Accumulate embeddings and generate pseudo-labels"""
    print(f"\n{'='*70}")
    print(f"PHASE 2: ACCUMULATION + PSEUDO-LABEL GENERATION")
    print(f"{'='*70}\n")
    
    model.eval()
    
    print("Step 1: Accumulating A1 from labeled data (correct predictions)...")
    with torch.no_grad():
        for i, (input, target, idx) in enumerate(labeledloader):
            input, target = input.cuda(), target.cuda()
            
            output, embeddings = model(input)
            predictions = output.argmax(dim=1)
            
            accumulator.add_labeled(embeddings, output, predictions, target, idx)
    
    print("\nStep 2: Accumulating A2 from unlabeled data (all samples)...")
    unlabeled_indices = []
    with torch.no_grad():
        for i, (uninput, untarget, unidx) in enumerate(unlabeledloader):
            uninput = uninput.cuda()
            
            unoutput, embeddings_unlabeled = model(uninput)
            accumulator.add_unlabeled(embeddings_unlabeled, unoutput, unidx)
            unlabeled_indices.extend(unidx.cpu().numpy() if torch.is_tensor(unidx) else unidx)
    
    acc_data = accumulator.get_accumulated_data()
    n_a1 = len(acc_data['correct_embeddings'])
    n_a2 = len(acc_data['unlabeled_embeddings'])
    print(f"\nTotal accumulated: A1={n_a1}, A2={n_a2}")
    
    if n_a1 == 0 or n_a2 == 0:
        print("WARNING: Insufficient data!")
        return
    
    print("\nStep 3: Generating GLOBAL pseudo-labels...")
    global_pseudo = generate_global_pseudo_labels_single_prototype(
        accumulator, num_classes=10
    )
    
    print("Step 4: Generating LOCAL pseudo-labels...")
    local_pseudo = generate_local_pseudo_labels_propagation(
        accumulator, k_neighbors=args.k_neighbors, alpha=args.alpha_propagation,
        num_classes=10, max_iter=50
    )
    
    if len(global_pseudo) > 0 and len(local_pseudo) > 0:
        pseudo_storage.update(unlabeled_indices, global_pseudo, local_pseudo)
        agreed_indices, _ = pseudo_storage.get_agreed_samples()
        global_rejections = len(global_pseudo) - len(agreed_indices)
        
        print(f"\nStatistics:")
        print(f"  Total unlabeled samples: {len(global_pseudo)}")
        print(f"  Rejections (no agreement): {global_rejections}")
    
    print(f"{'='*70}\n")

# =========================
# PHASE 3
# =========================
def phase3_semisupervised_training(unlabeledloader, model, optimizer, pseudo_storage, epoch):
    """Phase 3: Train with UNLABELED DATA ONLY using KLD_unlabeled"""
    print(f"\n{'='*70}")
    print(f"PHASE 3: SEMI-SUPERVISED TRAINING - Unlabeled Data Only")
    print(f"{'='*70}\n")
    
    model.train()
    
    batch_time = utils.AverageMeter()
    kld_unlabeled_losses = utils.AverageMeter()
    end = time.time()
    
    agreed_indices, agreed_labels = pseudo_storage.get_agreed_samples()
    print(f"Using {len(agreed_indices)} agreed samples\n")

    match_count_total = 0
    
    for i, (uninput, untarget, unidx) in enumerate(unlabeledloader):
        match_count = 0 
        uninput = uninput.cuda()
        
        unoutput, _ = model(uninput)
        
        kld_unlab, match_count = compute_kld_unlabeled(
            unoutput, unidx, pseudo_storage, num_classes=10, high=args.confidence_high
        )
        match_count_total = match_count_total + match_count
        
        loss = kld_unlab
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        kld_unlabeled_losses.update(kld_unlab.item(), uninput.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}] PHASE3\t'
                  'Time {batch_time.val:.3f}\t'
                  'KLD_U {kld_unlab.val:.4f} ({kld_unlab.avg:.4f})'.format(
                   epoch, i, len(unlabeledloader), batch_time=batch_time, kld_unlab=kld_unlabeled_losses))
    
    print("MATCH COUNT:", match_count_total)
    return kld_unlabeled_losses.avg, match_count_total

# =========================
# Statistics
# =========================
def compute_pseudo_label_statistics(pseudo_storage, true_labels):
    """Compute detailed statistics for pseudo-labels vs true labels"""
    total_samples = len(true_labels)

    global_correct = 0
    local_correct = 0
    both_correct = 0
    agree_but_wrong = 0
    disagree = 0

    for idx in range(len(true_labels)):
        actual_idx = idx + args.label_size
        true_label = true_labels[idx]

        if actual_idx in pseudo_storage.global_pseudo and actual_idx in pseudo_storage.local_pseudo:
            global_pred = pseudo_storage.global_pseudo[actual_idx]
            local_pred = pseudo_storage.local_pseudo[actual_idx]
            global_is_correct = (global_pred == true_label)
            local_is_correct = (local_pred == true_label)

            if global_is_correct:
                global_correct += 1
            if local_is_correct:
                local_correct += 1

            if global_pred == local_pred:
                if global_is_correct and local_is_correct:
                    both_correct += 1
                else:
                    agree_but_wrong += 1
            else:
                disagree += 1

    return {
        'total': total_samples,
        'global_correct': global_correct,
        'local_correct': local_correct,
        'both_correct': both_correct,
        'agree_but_wrong': agree_but_wrong,
        'disagree': disagree,
        'global_accuracy': 100.0 * global_correct / total_samples if total_samples > 0 else 0.0,
        'local_accuracy': 100.0 * local_correct / total_samples if total_samples > 0 else 0.0,
        'both_correct_rate': 100.0 * both_correct / total_samples if total_samples > 0 else 0.0,
        'agree_but_wrong_rate': 100.0 * agree_but_wrong / total_samples if total_samples > 0 else 0.0,
        'disagreement_rate': 100.0 * disagree / total_samples if total_samples > 0 else 0.0
    }

# =========================
# Main
# =========================
def main():
    # SET SEED FIRST for reproducibility
    set_seed(42)
    
    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # Note: cudnn settings are now in set_seed()

    # Save path
    save_path = args.save_folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Model
    num_class = 10
    model_dict = {"num_classes": num_class}
    base_model = resnet.resnet110(**model_dict).cuda()
    model = ResNetWrapper(base_model).cuda()

    # Optimizer & Scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=0.0001, nesterov=False)
    scheduler = MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

    # Logger
    train_logger = utils.Logger(os.path.join(save_path, 'train.log'))
    result_logger = utils.Logger(os.path.join(save_path, 'result.log'))

    # Get true labels for unlabeled data (for statistics only)
    true_unlabeled_labels = unlabeled_trainset.label_file

    print("\n" + "="*70)
    print("TRAINING START - 3-PHASE CORRECTED VERSION")
    print(f"Label size: {args.label_size}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print("="*70 + "\n")

    # Training loop
    for epoch in range(1, 301):
        scheduler.step()

        use_unlabeled = (epoch > args.warmup_epochs)

        if use_unlabeled:
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch}: 3-PHASE TRAINING")
            print(f"{'='*70}\n")

            accumulator = EmbeddingAccumulator()
            pseudo_storage = PseudoLabelStorage()

            loss_avg, ce_avg, kld_lab_avg, acc_avg = phase1_supervised_training(
                trainloader, model, optimizer, epoch
            )

            phase2_accumulation_and_pseudo_labels(
                trainloader, unlabeled_loader, model, accumulator, pseudo_storage, epoch
            )

            kld_unlab_avg, match_count_total = phase3_semisupervised_training(
                unlabeled_loader, model, optimizer, pseudo_storage, epoch
            )

            train_logger.write([epoch, loss_avg, ce_avg, kld_lab_avg, kld_unlab_avg, acc_avg])

            if len(pseudo_storage.global_pseudo) > 0:
                stats = compute_pseudo_label_statistics(pseudo_storage, true_unlabeled_labels)
                print(f"\n{'='*70}")
                print(f"EPOCH {epoch}: PSEUDO-LABEL STATISTICS")
                print(f"{'='*70}")
                print("FINAL MATCH COUNT L = G = Pred", match_count_total)
                print(f"Total: {stats['total']}")
                print(f"global_correct: {stats['global_correct']}")
                print(f"local_correct: {stats['local_correct']}")
                print(f"both_correct: {stats['both_correct']}")
                print(f"Agree but Wrong: {stats['agree_but_wrong']}")
                print(f"Disagreement: {stats['disagree']}")
                print(f"Global Accuracy: {stats['global_accuracy']:.2f}%")
                print(f"Local Accuracy: {stats['local_accuracy']:.2f}%")
                print(f"Both Correct: {stats['both_correct_rate']:.2f}%")
                print(f"Agree but Wrong: {stats['agree_but_wrong_rate']:.2f}%")
                print(f"Disagreement: {stats['disagreement_rate']:.2f}%")
                print(f"{'='*70}\n")

        else:
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch}: WARMUP - SUPERVISED ONLY")
            print(f"{'='*70}\n")

            loss_avg, ce_avg, kld_lab_avg, acc_avg = phase1_supervised_training(
                trainloader, model, optimizer, epoch
            )

            train_logger.write([epoch, loss_avg, ce_avg, kld_lab_avg, 0.0, acc_avg])

        # Evaluation
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}: EVALUATION")
        print(f"{'='*70}")
        acc, aurc, eaurc, aupr, fpr, ece, nll, brier = calc_metrics(testloader, model)
        print(f'\nSummary:')
        print(f'  Accuracy: {acc*100:.2f}%')
        print(f'  AURC: {aurc*1000:.2f}')
        print(f'  E-AURC: {eaurc*1000:.2f}')
        print(f'  AUPR: {aupr*100:.2f}')
        print(f'  FPR-95: {fpr*100:.2f}')
        print(f'  ECE: {ece*100:.2f}')
        print(f'  NLL: {nll*10:.2f}')
        print(f'  Brier: {brier*100:.2f}')
        print(f"{'='*70}\n")

        result_logger.write([epoch, acc, aurc*1000, eaurc*1000, aupr*100, fpr*100, ece*100, nll*10, brier*100])

        if epoch % 30 == 0:
            torch.save(model.state_dict(),
                      os.path.join(save_path, 'model_' + str(epoch) + '.pth'))

        if epoch == 300:
            torch.save(model.state_dict(),
                      os.path.join(save_path, 'model.pth'))

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70 + "\n")
    
if __name__ == "__main__":
    main()