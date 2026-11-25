# =============================================================================
# loss.py  —  Scale-Invariant Logarithmic loss (SiLog), Eigen et al. 2014
# -----------------------------------------------------------------------------
# Used to fine-tune the METRIC depth model (train.py).
#   d_i  = log(gt_i) - log(pred_i)                  (error in log space, per valid pixel)
#   loss = sqrt( mean(d_i^2) - lambda * (mean(d_i))^2 )
#
# Intuition:
#   * log space -> errors are RELATIVE (1 m off at 2 m is worse than 1 m off at 50 m)
#   * subtracting lambda*(mean d)^2 removes part of the penalty for a global scale
#     error: if pred is off by the same factor everywhere, every d_i is the same
#     constant. lambda = 1 would ignore global scale completely; lambda = 0.5 balances
#     "correct overall scale" (needed for metres) against "correct relative structure".
#
# (The paper's relative-depth model used different losses, L_ssi + L_gm, which are
#  not in this repo.)
# =============================================================================
import torch
from torch import nn


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        # only pixels with valid ground truth contribute
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss
