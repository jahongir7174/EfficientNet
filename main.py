import argparse
import copy
import csv
import os
import random
import warnings

import numpy
import torch
import tqdm
from torch.utils import data
from torchvision import transforms

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")
data_dir = os.path.join('/Projects', 'Dataset', 'IMAGENET')


def set_seed():
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def lr(args):
    base = 0.256 / 4096
    if not args.distributed:
        return args.batch_size * base
    else:
        return args.batch_size * args.world_size * base


def batch(images, target, model, criterion, is_train):
    images = images.cuda()
    target = target.cuda()
    if is_train:
        with torch.cuda.amp.autocast():
            loss = criterion(model(images), target)
        return loss
    else:
        pred = model(images)
        loss = criterion(model(images), target)
        acc1, acc5 = util.accuracy(pred, target, top_k=(1, 5))
        return loss, acc1, acc5


def train(args):
    model = nn.EfficientNet().cuda()
    amp_scale = torch.cuda.amp.GradScaler()
    optimizer = nn.RMSprop(util.add_weight_decay(model), lr(args), 0.9, 1e-3, 0, 0.9)
    ema = nn.EMA(model)
    if not args.distributed:
        model = torch.nn.parallel.DataParallel(model)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, [args.local_rank])
    scheduler = nn.StepLR(optimizer)
    criterion = nn.CrossEntropyLoss().cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    sampler = None
    dataset = Dataset(os.path.join(data_dir, 'train'),
                      transforms.Compose([util.Resize(args.input_size),
                                          util.RandomAugment(mean=9, n=2),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), normalize]))
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, not args.distributed,
                             sampler=sampler, num_workers=8, pin_memory=True)
    with open(f'weights/step.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'acc@1', 'acc@5', 'train_loss', 'val_loss'])
            writer.writeheader()
        best = 0
        for epoch in range(0, args.epochs):
            if args.distributed:
                sampler.set_epoch(epoch)
            p_bar = loader
            if args.local_rank == 0:
                print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
                p_bar = tqdm.tqdm(loader, total=len(loader))
            model.train()
            m_loss = util.AverageMeter()
            for images, target in p_bar:
                loss = batch(images, target, model, criterion, True)
                optimizer.zero_grad()
                amp_scale.scale(loss).backward()
                amp_scale.step(optimizer)
                amp_scale.update()
                ema.update(model)
                torch.cuda.synchronize()
                if not args.distributed:
                    loss = loss.item()
                    m_loss.update(loss, images.size(0))
                else:
                    loss = loss.data.clone()
                    torch.distributed.all_reduce(loss)
                    loss /= args.world_size
                    m_loss.update(loss, images.size(0))
                if args.local_rank == 0:
                    desc = ('%10s' + '%10.3g') % ('%g/%g' % (epoch + 1, args.epochs), loss)
                    p_bar.set_description(desc)

            scheduler.step(epoch + 1)
            if args.local_rank == 0:
                val_loss, acc1, acc5 = test(args, ema.model.eval())
                writer.writerow({'acc@1': str(f'{acc1:.3f}'),
                                 'acc@5': str(f'{acc5:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3),
                                 'val_loss': str(f'{val_loss.avg:.3f}'),
                                 'train_loss': str(f'{m_loss.avg:.3f}')})
                state = {'model': copy.deepcopy(ema.model).half()}
                torch.save(state, f'weights/last.pt')
                if acc1 > best:
                    torch.save(state, f'weights/best.pt')
                del state
                best = max(acc1, best)
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


def test(args, model=None):
    if model is None:
        model = torch.load('weights/best.pt', 'cuda')['model'].float().fuse()
        model.eval()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = Dataset(os.path.join(data_dir, 'val'),
                      transforms.Compose([transforms.Resize(args.input_size + 32),
                                          transforms.CenterCrop(args.input_size),
                                          transforms.ToTensor(), normalize]))

    loader = data.DataLoader(dataset, 64, num_workers=8, pin_memory=True)
    top1 = util.AverageMeter()
    top5 = util.AverageMeter()
    m_loss = util.AverageMeter()
    with torch.no_grad():
        for images, target in tqdm.tqdm(loader, ('%10s' * 2) % ('acc@1', 'acc@5')):
            loss, acc1, acc5 = batch(images, target, model, criterion, False)
            torch.cuda.synchronize()
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            m_loss.update(loss.item(), images.size(0))
        acc1, acc5 = top1.avg, top5.avg
        print('%10.3g' * 2 % (acc1, acc5))
    if model is None:
        torch.cuda.empty_cache()
    else:
        return m_loss, acc1, acc5


def profile(args):
    model = nn.EfficientNet().export().eval()
    shape = (1, 3, args.input_size, args.input_size)

    params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {int(params)}')
    if args.benchmark:
        util.print_benchmark(model, shape)


def main():
    set_seed()

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--epochs', default=450, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.world_size = int(os.getenv('WORLD_SIZE', 1))
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')
    if args.local_rank == 0:
        profile(args)
    if args.train:
        train(args)
    if args.test:
        test(args)


if __name__ == '__main__':
    main()
