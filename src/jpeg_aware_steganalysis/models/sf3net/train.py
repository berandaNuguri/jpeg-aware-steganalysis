import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import logging
from .sf3net import SF3Net
from .utils import train
from .Custom_Dataloader import return_dataloader


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',    # 1e-2
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0.001, metavar='wd',   # 코드 초기값 0.0005
                    help='weight_decay (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--start_nopc_epoch', type=int, default=10, help='start no pair constraint epoch')
parser.add_argument('--gpu', type=int, default=3, help='gpu device id')
parser.add_argument('--use_multi_gpu', type=bool, default=False, help='Use_multi_gpu')
parser.add_argument('--work_name', type=str, default='SF3Net-JUNIWARD-0.4-QF75', help='current work name')
parser.add_argument('--save_path', type=str, default='./checkpoints/sf3net/', help='save_path')
parser.add_argument('--load_model', type=bool, default=False, help='load_model')
parser.add_argument('--total_epochs', type=int, default=180, help='total_epoch')
parser.add_argument('--use_scheduler', action='store_true', help='If use this arg, Using MultistepLR Scheduler(80, 120, 140)')
parser.add_argument('--load_model_path', type=str, default='', help='load_model_path')
parser.add_argument('--csv_root', type=str, default='./csv/')
parser.add_argument('--data', type=str, default='/BOSSbase_BOWS2/R256_QF75/')
parser.add_argument('--steg_algo', type=str, default='J-UNIWARD_0.4')
args = parser.parse_args()
print(args.work_name)

torch.cuda.set_device(args.gpu)

#cuda related
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    args.gpu = None

Log_dir = args.save_path
if args.work_name != '':
    Log_dir = os.path.join(Log_dir, time.strftime('%Y%m%d-') + args.work_name)
else:
    Log_dir = os.path.join(Log_dir, time.strftime('%Y%m%d-') + 'train')
model_path = os.path.join(Log_dir, 'models')
os.makedirs(model_path, exist_ok=True)
log_dir = os.path.join(Log_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# set logger
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

train_file = f'{args.csv_root}/{args.data}/{args.steg_algo}_train.csv'
valid_file = f'{args.csv_root}/{args.data}/{args.steg_algo}_valid.csv'
test_file = f'{args.csv_root}/{args.data}/{args.steg_algo}_test.csv'

train_dataset = return_dataloader(train_file, 'train', rot=True)
valid_dataset = return_dataloader(valid_file, 'valid', rot=False)
test_dataset = return_dataloader(test_file, 'test', rot=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

my_model = SF3Net()
def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')
    if type(module) == nn.Linear:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

my_model.apply(initWeights)

params = my_model.parameters()
params_wd, params_rest = [], []
for param_item in params:
    if param_item.requires_grad:
        (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

param_groups = [{'params': params_wd, 'weight_decay': args.weight_decay}, {'params': params_rest}]
optimizer = optim.SGD(param_groups, lr=args.lr, momentum=args.momentum)

if args.use_scheduler:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 140], gamma=0.1)
else:
    scheduler = None

# load model
if args.load_model:
    ckpt = torch.load(args.load_model_path, map_location='cpu')
    my_model.load_state_dict(ckpt)

my_model.cuda()
criterion = nn.CrossEntropyLoss()

if args.start_nopc_epoch is None:
    args.start_nopc_epoch = args.total_epochs
logging.info("Prepare to training %d epochs PC + %d epochs noPC !"
             % (args.start_nopc_epoch, args.total_epochs - args.start_nopc_epoch))

train(my_model, optimizer, scheduler, criterion, train_loader, valid_loader, test_loader,  model_path,
      log_dir, logging, args)