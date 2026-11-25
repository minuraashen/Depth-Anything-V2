# =============================================================================
# metric.py  —  standard depth-estimation evaluation metrics
# -----------------------------------------------------------------------------
# pred, target: 1D tensors of depth values at the valid pixels only (metres).
# These are the same metrics reported in the paper's tables (e.g. Table 4: d1, AbsRel, RMSE).
# =============================================================================
import torch


def eval_depth(pred, target):
    assert pred.shape == target.shape

    # Ratio between prediction and GT, always >= 1 (1 = perfect)
    thresh = torch.max((target / pred), (pred / target))

    # d1 / d2 / d3 ("delta" accuracy): fraction of pixels whose ratio is below
    # 1.25, 1.25^2, 1.25^3  -> HIGHER is better (d1 is the one usually quoted)
    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    # Error metrics -> LOWER is better
    abs_rel = torch.mean(torch.abs(diff) / target)          # AbsRel: mean |pred - gt| / gt  (relative error)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)        # SqRel

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))       # RMSE in metres
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))   # same formula as the loss

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}