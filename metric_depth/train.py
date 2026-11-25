# =============================================================================
# metric_depth/train.py  —  FINE-TUNING Depth Anything V2 for METRIC depth (metres)
# -----------------------------------------------------------------------------
# This is the ONLY training code released in the repo. It is NOT the training of the
# main relative-depth model from the paper (synthetic-only teacher -> 62M pseudo
# labels -> students with L_ssi + L_gm); that code was not released.
#
# What it does (paper Sec. 7.3 / App. B.12):
#   * start from the released RELATIVE-depth checkpoint (encoder weights only)
#   * fine-tune on a SYNTHETIC dataset with metric ground truth:
#       Hypersim        (indoor,  max_depth = 20 m)
#       Virtual KITTI 2 (outdoor, max_depth = 80 m)
#   * loss: scale-invariant log loss (SiLog), see util/loss.py
#   * evaluate every epoch with standard depth metrics (d1, AbsRel, RMSE, ...)
# Runs on multiple GPUs with PyTorch DistributedDataParallel; launch with dist_train.sh.
# =============================================================================
import argparse
import logging
import os
import pprint
import random

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from depth_anything_v2.dpt import DepthAnythingV2   # the METRIC version (Sigmoid x max_depth output)
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log


parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='hypersim', choices=['hypersim', 'vkitti'])
parser.add_argument('--img-size', default=518, type=int)          # training crop size 518 x 518
parser.add_argument('--min-depth', default=0.001, type=float)     # ignore GT depth outside [min, max]
parser.add_argument('--max-depth', default=20, type=float)        # 20 m indoor, 80 m outdoor
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=2, type=int)                  # batch size PER GPU
parser.add_argument('--lr', default=0.000005, type=float)         # 5e-6 for the encoder (decoder gets 10x)
parser.add_argument('--pretrained-from', type=str)                # path to the relative-depth checkpoint
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()
    
    warnings.simplefilter('ignore', np.RankWarning)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    # Multi-GPU setup: rank = id of this process/GPU, world_size = number of GPUs
    rank, world_size = setup_distributed(port=args.port)
    
    if rank == 0:   # only the main process logs and writes TensorBoard files
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
    
    cudnn.enabled = True
    cudnn.benchmark = True   # let cuDNN pick the fastest conv algorithms (input size is fixed)
    
    # ---------------- Datasets ----------------
    # Train on synthetic data with exact metric depth; validate on Hypersim val (indoor)
    # or on real KITTI (outdoor) -> tests synthetic-to-real transfer.
    size = (args.img_size, args.img_size)
    if args.dataset == 'hypersim':
        trainset = Hypersim('dataset/splits/hypersim/train.txt', 'train', size=size)
    elif args.dataset == 'vkitti':
        trainset = VKITTI2('dataset/splits/vkitti2/train.txt', 'train', size=size)
    else:
        raise NotImplementedError
    # DistributedSampler: each GPU gets a different slice of the data each epoch
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler)
    
    if args.dataset == 'hypersim':
        valset = Hypersim('dataset/splits/hypersim/val.txt', 'val', size=size)
    elif args.dataset == 'vkitti':
        valset = KITTI('dataset/splits/kitti/val.txt', 'val', size=size)
    else:
        raise NotImplementedError
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, sampler=valsampler)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # ---------------- Model ----------------
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    
    if args.pretrained_from:
        # Load ONLY the encoder weights (parameter names containing 'pretrained' = the DINOv2
        # encoder) from the relative-depth model. The DPT decoder starts from scratch, because
        # its output meaning changes (inverse depth -> metric depth).
        model.load_state_dict({k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k}, strict=False)
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)   # BN statistics shared across GPUs
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)
    
    criterion = SiLogLoss().cuda(local_rank)   # scale-invariant log loss (util/loss.py)
    
    # Two learning rates (same 1:10 ratio as the paper, Sec. 7.1):
    #   encoder (already well trained)  -> small lr  (5e-6), so it is only gently adjusted
    #   decoder (new / task-specific)   -> 10x larger (5e-5)
    optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
                      lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    total_iters = args.epochs * len(trainloader)
    
    # Best value seen so far for each metric (higher is better for d1-d3, lower for the rest)
    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}
    
    for epoch in range(args.epochs):
        if rank == 0:
            logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs, previous_best['d1'], previous_best['d2'], previous_best['d3']))
            logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
                        'log10: {:.3f}, silog: {:.3f}'.format(
                            epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'], 
                            previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))
        
        trainloader.sampler.set_epoch(epoch + 1)   # reshuffle differently every epoch
        
        # ======================= TRAINING =======================
        model.train()
        total_loss = 0
        
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            
            img, depth, valid_mask = sample['image'].cuda(), sample['depth'].cuda(), sample['valid_mask'].cuda()
            
            # Data augmentation: random horizontal flip (image, GT and mask together)
            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)
            
            pred = model(img)   # [B, H, W] predicted depth in metres
            
            # Loss only on valid pixels whose GT lies inside [min_depth, max_depth]
            loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))
            
            loss.backward()     # backpropagation
            optimizer.step()    # update weights
            
            total_loss += loss.item()
            
            iters = epoch * len(trainloader) + i
            
            # "Poly" learning-rate schedule: decays smoothly from lr to 0 over training
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            
            optimizer.param_groups[0]["lr"] = lr            # encoder
            optimizer.param_groups[1]["lr"] = lr * 10.0     # decoder (kept at 10x)
            
            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)
            
            if rank == 0 and i % 100 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))
        
        # ======================= VALIDATION =======================
        model.eval()
        
        results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 
                   'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
                   'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
        nsamples = torch.tensor([0.0]).cuda()
        
        for i, sample in enumerate(valloader):
            
            img, depth, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]
            
            with torch.no_grad():
                pred = model(img)
                # val images are not resized together with their GT, so resize the
                # prediction to the GT resolution before comparing
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
            
            valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
            
            if valid_mask.sum() < 10:
                continue   # skip images with almost no valid GT
            
            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])   # util/metric.py
            
            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1
        
        torch.distributed.barrier()   # wait for all GPUs
        
        # Sum the metric totals from all GPUs onto GPU 0, then average
        for k in results.keys():
            dist.reduce(results[k], dst=0)
        dist.reduce(nsamples, dst=0)
        
        if rank == 0:
            logger.info('==========================================================================================')
            logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
            logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
            logger.info('==========================================================================================')
            print()
            
            for name, metric in results.items():
                writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)
        
        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())
        
        # Save the latest model + optimizer state every epoch (overwrites latest.pth)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))


if __name__ == '__main__':
    main()