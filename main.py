import argparse
import os
import pdb
import random
import shutil
import time
import warnings
import copy
import json
from enum import Enum

import itertools
import torch
import deepspeed
import wandb
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset, ConcatDataset
from sklearn.metrics import roc_auc_score

from hyperbox.mutator import OnehotMutator, RandomMutator

from ham10000_datamodule import Ham10000Dataset
from vit import VisionTransformer, ViT_B, ViT_L

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
custom_model_names = {
    'visiontransformer': VisionTransformer,
    'vit_b': ViT_B,
    'vit_l': ViT_L
}


def add_argument():
    parser = argparse.ArgumentParser(description='MedPipe Training')
    parser.add_argument('--data', metavar='DIR', type=str, default='/data2/share/skin_cancer_data',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('--log_dir', metavar='DIR', type=str,
                        help='log_dir')

    ######### model
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT_B',
                        choices=model_names+list(custom_model_names.keys()),
                        help='model architecture: ' +
                            ' | '.join(model_names+list(custom_model_names.keys())) +
                            ' (default: ViT_B)')
    parser.add_argument('--num_classes', type=int, default=7, help="number of classes")

    ######### mutator
    parser.add_argument('--search', action='store_true', help="conduct search")
    parser.add_argument('--unrolled', action='store_true', help="unrolled gradients")
    parser.add_argument('--mutator', default='OnehotMutator', type=str,
                        help='mutator type')
    parser.add_argument('--arch_path', type=str, default=None,
                        help='the path of the searched architecture (json file)')

    ######### dataset 
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
    parser.add_argument('--concat_train_val', action='store_true', help="concat train and val datasets")
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    
    ######### training 
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    ######### distributed 
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    parser.add_argument('--local_rank', type=int, default=-1, help="local rank for distributed training on gpus")
    parser.add_argument('--ipdb_debug', action='store_true', help="enable ipdb debug")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

best_acc1 = 0
best_auc = 0

def main():
    args = add_argument()
    print(f"rank{args.local_rank} args:{args}")
    if args.ipdb_debug:
        from ipdb import set_trace
        set_trace()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1

    args.world_size = ngpus_per_node * args.world_size
    t_losses, t_acc1s = main_worker(args.gpu, ngpus_per_node, args)
    #dist.barrier()
    
    # Write the losses to an excel file
    if dist.get_rank() ==0:
        all_losses = [torch.empty_like(t_losses) for _ in range(ngpus_per_node)]
        dist.gather(tensor=t_losses, gather_list=all_losses,dst=0)
    else:
        dist.gather(tensor=t_losses, dst=0)

    if dist.get_rank() ==0:
        all_acc1s = [torch.empty_like(t_acc1s) for _ in range(ngpus_per_node)]
        dist.gather(tensor=t_acc1s, gather_list=all_acc1s,dst=0)
    else:
        dist.gather(tensor=t_acc1s, dst=0)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print(f"rank{args.local_rank} Use GPU: {args.gpu} for training")

    if args.arch in model_names:
        if args.pretrained:
            print(f"rank{args.local_rank}=> using pre-trained model '{args.arch}'")
            model = models.__dict__[args.arch](pretrained=True, num_classes=args.num_classes)
        else:
            print(f"rank{args.local_rank}=> creating model '{args.arch}'")
            model = models.__dict__[args.arch](num_classes=args.num_classes)
    elif args.arch.lower() in custom_model_names:
        model = custom_model_names[args.arch](image_size=(448,608), num_classes=args.num_classes)

    mutator = None
    optimizer_mutator = None
    if args.search:
        mutator = OnehotMutator(model)
        optimizer_mutator = torch.optim.Adam(
            mutator.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=1.0E-3)

    if args.arch_path:
        os.system(f"cp {args.arch_path} {args.log_dir}/") # backup the arch json file
        mutator = OnehotMutator(model)
        with open(args.arch_path, 'r') as f:
            mask = json.load(f)
        mutator.sample_by_mask(mask)

    # In case of distributed process, initializes the distributed backend
    # which will take care of sychronizing nodes/GPUs
    if args.local_rank == -1:
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()

    args.batch_size = int(args.batch_size / ngpus_per_node)
    if not torch.cuda.is_available():# and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
        device = torch.device("cpu")
        model = model.to(device)

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    # Initialize DeepSpeed for the model
    model, optimizer, _, _ = deepspeed.initialize(
        model = model, optimizer = optimizer, args = args,
        lr_scheduler = None,#scheduler,
        dist_init_required=True
    )
    if args.search:
        mutator, optimizer_mutator, _, _ = deepspeed.initialize(
            args=args, model=mutator, model_parameters=mutator.parameters(), optimizer=optimizer_mutator)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"rank{args.local_rank}=> loading checkpoint '{args.resume}'")
            # # 使用DeepSpeedEngine读取 checkpoint
            # tag = 'best' # or 'last
            # load_path, checkpoint = model.load_checkpoint(args.resume, tag='best')
            # load_path, checkpoint = mutator.load_checkpoint(args.resume, tag='best') # failed
            # args.start_epoch = checkpoint['epoch']
            # best_acc1 = checkpoint['best_acc1']
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(args.gpu)

            # 使用 pytorch 读取 checkpoint
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if args.search:
                mutator.load_state_dict(checkpoint['mutator'])
                optimizer_mutator.load_state_dict(checkpoint['optimizer_mutator'])
            print(f"rank{args.local_rank}=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"rank{args.local_rank}=> no checkpoint found at '{args.resume}'")


    # Data loading code
    if args.dummy:
        print(f"rank{args.local_rank}=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'skin_cancer_train')
        valdir = os.path.join(args.data, 'skin_cancer_val')
        testdir = os.path.join(args.data, 'skin_cancer_test')
        train_dataset = Ham10000Dataset(traindir, add_random_transforms=True)
        val_dataset = Ham10000Dataset(valdir)
        test_dataset = Ham10000Dataset(testdir)
            

    if args.local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    print(f"rank{args.local_rank} Batch_size:{args.batch_size}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)


    if args.evaluate:
        if mutator is not None:
            mutator.reset()
        loss = torch.empty(1).cuda()
        acc1 = torch.empty(1).cuda()
        metrics_dict = validate(test_loader, model, criterion, args)
        loss[0] = metrics_dict['loss'].avg
        acc1[0] = metrics_dict['top1'].avg
        # auc = metrics_dict['auc'].avg
        # print(f'Accuracy: {acc1:.4f} AUC:{auc:.4f}')
        print(f'Accuracy: {acc1} ')
        return (loss, acc1)

    losses = torch.empty(args.epochs).cuda()
    acc1s = torch.empty(args.epochs).cuda()
    # aucs = torch.empty(args.epochs).cuda()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        if args.search:
            this_loss = search(train_loader, val_loader, model, mutator, criterion, optimizer, optimizer_mutator, epoch, device, args)
        else:
            this_loss = train(train_loader, model, criterion, optimizer, epoch, device, args)
        losses[epoch] = this_loss

        # evaluate on validation set
        metrics_dict = validate(val_loader, model, criterion, args)
        acc1s[epoch] = metrics_dict['top1'].avg
        # aucs[epoch] = metrics_dict['auc'].avg
        predictions = metrics_dict['predictions']
        labels = metrics_dict['labels']

        scheduler.step()
        
        # remember best acc@1 and save checkpoint
        acc1 = acc1s[epoch]
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.gpu is None):
            ckpt = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                # 'auc': aucs[epoch].item(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'labels': labels,
                'predictions': predictions
            }
            if args.search:
                ckpt['mutator'] = mutator.state_dict()
                ckpt['optimizer_mutator'] = optimizer_mutator.state_dict()
            save_checkpoint(ckpt, is_best, args.log_dir, model)
        if args.search:
            arch_save_path = f'{args.log_dir}/archs'
            os.system(f"mkdir -p {arch_save_path}")
            mutator.save_arch(f'{arch_save_path}/arch_{epoch}_acc{acc1}.json')

    return (losses, acc1s)
    # return (losses, acc1s, aucs)


######## darts search related
def _compute_virtual_model(X, y, model, optimizer, criterion, mutator, lr, momentum, weight_decay):
    """
    Compute unrolled weights w`
    """
    # don't need zero_grad, using autograd to calculate gradients
    _, loss = _logits_and_loss(X, y, model, criterion, mutator)
    gradients = torch.autograd.grad(loss, model.parameters())
    with torch.no_grad():
        for w, g in zip(model.parameters(), gradients):
            m = optimizer.state[w].get("momentum_buffer", 0.)
            w = w - lr * (momentum * m + g + weight_decay * w)

def _compute_hessian(backup_params, dw, trn_X, trn_y, model, mutator, criterion):
    """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
    """
    _restore_weights(model, backup_params)
    norm = torch.cat([w.view(-1) for w in dw]).norm()
    eps = 0.01 / norm
    if norm < 1E-8:
        print(
            "In computing hessian, norm is smaller than 1E-8, cause eps to be %.6f.", norm.item())

    dalphas = []
    for e in [eps, -2. * eps]:
        # w+ = w + eps*dw`, w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(model.parameters(), dw):
                p += e * d

        _, loss = _logits_and_loss(trn_X, trn_y, model, criterion, mutator)
        dalphas.append(torch.autograd.grad(loss, mutator.parameters()))

    dalpha_pos, dalpha_neg = dalphas  # dalpha { L_trn(w+) }, # dalpha { L_trn(w-) }
    hessian = [(p - n) / 2. * eps for p, n in zip(dalpha_pos, dalpha_neg)]
    return hessian

def _restore_weights(model, backup_params):
    with torch.no_grad():
        for param, backup in zip(model.parameters(), backup_params):
            param.copy_(backup)

def _unrolled_backward(trn_X, trn_y, val_X, val_y, model, optimizer, criterion, mutator):
    """
    Compute unrolled loss and backward its gradients
    """
    backup_params = copy.deepcopy(tuple(model.parameters()))

    # do virtual step on training data
    lr = optimizer.param_groups[0]["lr"]
    momentum = optimizer.param_groups[0]["momentum"]
    weight_decay = optimizer.param_groups[0]["weight_decay"]
    _compute_virtual_model(trn_X, trn_y, model, optimizer, criterion, mutator, lr, momentum, weight_decay)

    # calculate unrolled loss on validation data
    # keep gradients for model here for compute hessian
    _, loss = _logits_and_loss(val_X, val_y, model, criterion, mutator)
    w_model, w_ctrl = tuple(model.parameters()), tuple(mutator.parameters())
    w_grads = torch.autograd.grad(loss, w_model + w_ctrl)
    d_model, d_ctrl = w_grads[:len(w_model)], w_grads[len(w_model):]

    # compute hessian and final gradients
    hessian = _compute_hessian(backup_params, d_model, trn_X, trn_y, model, mutator, criterion)
    with torch.no_grad():
        for param, d, h in zip(w_ctrl, d_ctrl, hessian):
            # gradient = dalpha - lr * hessian
            param.grad = d - lr * h

    # restore weights
    _restore_weights(model, backup_params)

def search(
        train_loader,
        val_loader,
        model,
        mutator,
        criterion,
        optimizer,
        optimizer_mutator,
        epoch,
        device,
        args
    ):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    mutator.train()
    val_cycle_iter = itertools.cycle(val_loader)

    end = time.time()
    for i, (trn_X, trn_y) in enumerate(train_loader):
        if args.ipdb_debug and i == 10:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        (val_X, val_y) = next(val_cycle_iter)
        trn_X = trn_X.to(device, non_blocking=True)
        trn_y = trn_y.to(device, non_blocking=True)
        val_X = val_X.to(device, non_blocking=True)
        val_y = val_y.to(device, non_blocking=True)

        ###### phase 1. architecture step
        model.eval()
        mutator.train()
        mutator.zero_grad()
        if args.unrolled:
            _, val_loss = _unrolled_backward(trn_X, trn_y, val_X, val_y, model, optimizer, criterion, mutator) 
        else:
            _, val_loss = _logits_and_loss(val_X, val_y, model, criterion, mutator)
        mutator.backward(val_loss)
        mutator.step()

        ###### phase 2: child network step
        model.train()
        mutator.eval()
        model.zero_grad()
        output, loss = _logits_and_loss(trn_X, trn_y, model, criterion, None)
        model.backward(loss)
        model.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, trn_y, topk=(1, 5))
        losses.update(loss.item(), trn_X.size(0))
        top1.update(acc1[0], trn_X.size(0))
        top5.update(acc5[0], trn_X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    return (float(losses.val))

def _logits_and_loss(X, y, model, criterion, mutator=None):
    if mutator is not None:
        mutator.reset()
    output = model(X)
    loss = criterion(output, y)
    return output, loss

def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if args.ipdb_debug and i == 10:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        model.backward(loss)
        model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    return (float(losses.val))

def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        labels = []
        predictions = []
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                if args.ipdb_debug and i == 10:
                    break
                i = base_progress + i

                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)
                    images = images.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                labels.append(target)
                predictions.append(output)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                # auc = roc_auc(output, target, num_classes=images.shape(-1))
                losses.update(loss.item(), images.size(0))
                # aucs.update(np.mean(auc), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)
        predictions = torch.cat(predictions)
        labels = torch.cat(labels)
        return labels, predictions

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    # aucs = AverageMeter('AUC', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    # progress = ProgressMeter(
    #     len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
    #     [batch_time, losses, aucs, top1, top5],
    #     prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    labels, predictions = run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()
        # aucs.all_reduce()    

        labels_list = [torch.empty_like(labels) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(labels_list, labels)
        labels = torch.cat(labels_list)

        predictions_list = [torch.empty_like(predictions) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(predictions_list, predictions)
        predictions = torch.cat(predictions_list)

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return {
        'top1': top1,
        'top5': top5,
        # 'auc': aucs,
        'loss': losses,
        'labels': labels,
        'predictions': predictions
    }

def save_checkpoint(state, is_best, filepath, ds_engine=None):
    if ds_engine is not None:
        # 使用DeepSpeedEngine 保存checkpoint数据
        filename = f"{filepath}/checkpoints"
        tag = 'best' if is_best else 'last'
        ds_engine.save_checkpoint(filename, tag=tag, client_state=state)
    else:
        # 使用原始pytorch保存checkpoint数据
        filename = f"{filepath}/checkpoint.pth.tar"
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def roc_auc(output, target, num_classes=2):
    """Computes the ROC-AUC score for the specified number of classes"""
    with torch.no_grad():
        probs = torch.nn.functional.softmax(output, dim=1)
        probs = probs.cpu().numpy()
        target = target.cpu().numpy()

        # Compute ROC-AUC for each class
        auc_scores = []
        for i in range(num_classes):
            auc = roc_auc_score(target, probs[:, i], multi_class='ovr', labels=[i])
            auc_scores.append(auc)

        # Return average ROC-AUC score or individual scores for each class
        if num_classes == 1:
            return auc_scores[0]
        else:
            return auc_scores



if __name__ == '__main__':
    main()
