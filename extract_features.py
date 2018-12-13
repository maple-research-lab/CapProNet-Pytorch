from __future__ import division

import os, sys, pdb, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import models
import numpy as np

model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data_path', type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='resnet18', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
# Checkpoints
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--Ddim', type=int, default=4)
args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

if args.manualSeed is None:
  args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
  torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

def main():
  # Init logger
  if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
  log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
  print_log('save path : {}'.format(args.save_path), log)
  state = {k: v for k, v in args._get_kwargs()}
  print_log(state, log)
  print_log("Random Seed: {}".format(args.manualSeed), log)
  print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
  print_log("torch  version : {}".format(torch.__version__), log)
  print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

  # Init dataset
  if not os.path.isdir(args.data_path):
    os.makedirs(args.data_path)

  if args.dataset == 'cifar10':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
  elif args.dataset == 'cifar100':
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
  else:
    assert False, "Unknow dataset : {}".format(args.dataset)

  train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
     transforms.Normalize(mean, std)])
  test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])

  if args.dataset == 'cifar10':
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 10
  elif args.dataset == 'cifar100':
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 100
  elif args.dataset == 'svhn':
    train_data = dset.SVHN(args.data_path, split='train', transform=train_transform, download=True)
    test_data = dset.SVHN(args.data_path, split='test', transform=test_transform, download=True)
    num_classes = 10
  elif args.dataset == 'stl10':
    train_data = dset.STL10(args.data_path, split='train', transform=train_transform, download=True)
    test_data = dset.STL10(args.data_path, split='test', transform=test_transform, download=True)
    num_classes = 10
  elif args.dataset == 'imagenet':
    assert False, 'Do not finish imagenet code'
  else:
    assert False, 'Do not support dataset : {}'.format(args.dataset)

  train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

  print_log("=> creating model '{}'".format(args.arch), log)
  # Init model, criterion, and optimizer
  net = models.__dict__[args.arch](num_classes, args.Ddim)
  print_log("=> network :\n {}".format(net), log)

  net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

  if args.use_cuda:
    net.cuda()

  # optionally resume from a checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      print_log("=> loading checkpoint '{}'".format(args.resume), log)
      checkpoint = torch.load(args.resume)
      recorder = checkpoint['recorder']
      net.load_state_dict(checkpoint['state_dict'])
      print_log("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']), log)
    else:
      print_log("=> no checkpoint found at '{}'".format(args.resume), log)
  else:
    print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

  # Main loop
  
  val_acc = validate(train_loader, net, log, args.Ddim, args.save_path)
  print(val_acc)
  
#  for epoch in range(args.start_epoch, args.epochs):
#    # train for one epoch
#    train_acc, train_los = train(train_loader, net, criterion, optimizer, epoch, log, args.Ddim)
#
#    # evaluate on validation set
#    #val_acc,   val_los   = extract_features(test_loader, net, criterion, log)
#    val_acc,   val_los   = validate(test_loader, net, criterion, log, args.Ddim)
#    is_best = recorder.update(epoch, train_los, train_acc, val_los, val_acc)
#
#    save_checkpoint({
#      'epoch': epoch + 1,
#      'arch': args.arch,
#      'state_dict': net.state_dict(),
#      'recorder': recorder,
#      'optimizer' : optimizer.state_dict(),
#    }, is_best, args.save_path, 'checkpoint.pth.tar')
#
#    # measure elapsed time
#    epoch_time.update(time.time() - start_time)
#    start_time = time.time()
#    recorder.plot_curve( os.path.join(args.save_path, 'curve.png') )

  log.close()


def validate(val_loader, model, log, Ddim, save_path):
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  for i, (input, target) in enumerate(val_loader):
    eye = torch.eye(Ddim)
    if args.use_cuda:
      target = target.cuda(async=True)
      input = input.cuda()
      eye = eye.cuda()
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)
    eye_var = torch.autograd.Variable(eye, volatile=True)

    # compute output
    features, output = model(input_var, eye_var)
    if i == 0:
        features_data = features.cpu().data.numpy()
        target_data = target.cpu().numpy()
    else:
        features_data = np.concatenate((features_data, features.cpu().data.numpy()), axis = 0)
        target_data = np.concatenate((target_data, target.cpu().numpy()), axis = 0)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    top1.update(prec1[0], input.size(0))
    top5.update(prec5[0], input.size(0))

  np.save(save_path+'/features',features_data)
  np.save(save_path+'/labels',target_data)
  np.save(save_path+'/weights',model.module.classifier.weight.cpu().data.numpy())
  print(features_data.shape)
  print(target_data.shape)
  return top1.avg

def extract_features(val_loader, model, criterion, log):
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  for i, (input, target) in enumerate(val_loader):
    if args.use_cuda:
      target = target.cuda(async=True)
      input = input.cuda()
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    # compute output
    output, features = model([input_var])

    pdb.set_trace()

    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    losses.update(loss.data[0], input.size(0))
    top1.update(prec1[0], input.size(0))
    top5.update(prec5[0], input.size(0))

  print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

  return top1.avg, losses.avg

def print_log(print_string, log):
  print("{}".format(print_string))
  log.write('{}\n'.format(print_string))
  log.flush()

def save_checkpoint(state, is_best, save_path, filename):
  filename = os.path.join(save_path, filename)
  torch.save(state, filename)
  if is_best:
    bestname = os.path.join(save_path, 'model_best.pth.tar')
    shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = args.learning_rate
  assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
  for (gamma, step) in zip(gammas, schedule):
    if (epoch >= step):
      lr = lr * gamma
    else:
      break
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return lr

def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res

if __name__ == '__main__':
  main()
