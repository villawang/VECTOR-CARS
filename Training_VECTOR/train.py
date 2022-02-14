import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
import scipy.signal
from tensorboardX import SummaryWriter
import time
import pdb
import argparse

# self defined modules
from models import CAE
import utils

def loss_function(recon_x, x): #hidden_neurons_batch: [samples, number of neurons]
    # BCE = F.mse_loss(recon_x.view(-1, 1000), x.view(-1, 1000))
    BCE = F.l1_loss(recon_x.view(-1, 1000), x.view(-1, 1000))
    return BCE.cuda()

def adjust_learning_rate(learning_rate, optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = learning_rate * (0.1 ** (sum(epoch >= np.array(lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new


# windows might need to manually change in the function
def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=str, default='0')
    parser.add_argument('--is_train', default=False, action='store_true')
    parser.add_argument('--is_skip', default=False, action='store_true')
    parser.add_argument('--batch_size', '--bs', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_steps', type=float, default=[25,50,75], nargs="+",
                        help='lr steps for decreasing learning rate') # batch: 64 [10,20,30], [5,10,20]
    parser.add_argument('--base_model', default='cae_4', type=str)
    parser.add_argument('--dataset', default=9, type=int)
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
    args = parser.parse_args()
    return args

args = parse_opts()


print(args)


os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_id

# create tensorboard writer
cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))



if args.base_model == 'cae_4':
    model = CAE.CAE_4(data_len=1000, kernel_size=8, is_skip=args.is_skip)
elif args.base_model == 'cae_5':
    model = CAE.CAE_5(data_len=1000, kernel_size=8, is_skip=args.is_skip)
elif args.base_model == 'cae_6':
    model = CAE.CAE_6(data_len=1000, kernel_size=8, is_skip=args.is_skip)
elif args.base_model == 'cae_7':
    model = CAE.CAE_7(data_len=1000, kernel_size=8, is_skip=args.is_skip)
elif args.base_model == 'cae_8':
    model = CAE.CAE_8(data_len=1000, kernel_size=8, is_skip=args.is_skip)
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

from Generate_Data import *
class raman_dataset_fast(Dataset):
    def __init__(self, dataset, size):
        self.cars_data, self.raman_data = generate_datasets(dataset,size)
         
    def __len__(self):
        return len(self.raman_data)

    def __getitem__(self, idx):
        raman_data = self.raman_data[idx]
        cars_data = self.cars_data[idx]
        return raman_data, cars_data

class raman_dataset(Dataset):
    def __init__(self, file_path, raman_file, cars_file):
        self.raman_data = pd.read_csv(os.path.join(file_path, raman_file)).iloc[:, 1:]
        self.cars_data = pd.read_csv(os.path.join(file_path, cars_file)).iloc[:, 1:]
        
    def __len__(self):
        return len(self.raman_data)

    def __getitem__(self, idx):
        raman_data = self.raman_data.values[idx]
        cars_data = self.cars_data.values[idx]
        return raman_data, cars_data

# define model save path
if args.is_skip == True:
    model_save_dir = os.path.join('trained_model', '{}-skip'.format(args.base_model),'{}-dataset'.format(args.dataset))
    logdir = os.path.join('log', '{}-skip'.format(args.base_model),'{}-dataset'.format(args.dataset))
else:
    model_save_dir = os.path.join('trained_model', '{}-noskip'.format(args.base_model),'{}-dataset'.format(args.dataset))
    logdir = os.path.join('log', '{}-noskip'.format(args.base_model),'{}-dataset'.format(args.dataset))

print('Before if is train')

# training
if args.is_train:
    print('Loading dataset.....')
    dataset_train = raman_dataset_fast(args.dataset,10000)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dataset_val = raman_dataset_fast(args.dataset,2000)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)


    # training
    best_loss = 100. # check if best val loss to save model
    train_loss = utils.AverageMeter() # train loss
    val_loss = utils.AverageMeter() # validation loss
    for epoch in trange(args.epochs):
        model.train()
        train_loss.reset()
        val_loss.reset()

        for step, inputs in enumerate(train_loader):
            raman = inputs[0].float().cuda()
            cars = inputs[1].float().cuda()
            optimizer.zero_grad()
            outputs = model(cars)
            loss = loss_function(outputs, raman)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), raman.size(0))

            if step % 20 == 0:
                print('-------------------------------------------------------')
                print('lr: ', optimizer.param_groups[0]['lr'])
                print_string = 'XEpoch: [{0}][{1}/{2}], loss: {loss:.5f}'.format(epoch, step, len(train_loader), loss=train_loss.avg)
                print(print_string)
            
        # validation
        model.eval()
        with torch.no_grad():
            for val_step, inputs in enumerate(val_loader):
                raman = inputs[0].float().cuda()
                cars = inputs[1].float().cuda()
                outputs = model(cars)
                loss_valid = loss_function(outputs, raman)
                val_loss.update(loss_valid.item(), raman.size(0))
        print('----validation----')
        print_string = 'yEpoch: [{0}][{1}/{2}], loss: {loss:.5f}'.format(epoch, val_step, len(val_loader), loss=val_loss.avg)
        print(print_string)
        
        # adjust learning rate 
        adjust_learning_rate(args.lr, optimizer, epoch, args.lr_steps)
        writer.add_scalar('train_loss', train_loss.avg, epoch)
        writer.add_scalar('val_loss', val_loss.avg, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        # save the best model
        if val_loss.avg < best_loss:
            checkpoint = os.path.join(model_save_dir, 'checkpoint'+str(args.dataset)+'.pth.tar')
            utils.save_checkpoint(model, optimizer, epoch, checkpoint)        
            best_loss = val_loss.avg
        print('Best loss: {:.5f}'.format(best_loss))
    print('Finished Training/Validation')
else: # testing
    print('Loading dataset.....')
    if args.dataset == 1:
        a=1
        b='a'
    elif args.dataset == 2:
        a=1
        b='b'
    elif args.dataset == 3:
        a=1
        b='c'
    elif args.dataset == 4:
        a=2
        b='a'
    elif args.dataset == 5:
        a=2
        b='b'
    elif args.dataset == 6:
        a=2
        b='c'
    elif args.dataset == 7:
        a=3
        b='a'
    elif args.dataset == 8:
        a=3
        b='b'
    else:
        a=3
        b='c'
    #dataset_val = raman_dataset('data', str(a)+b+'Raman_spectrums_valid.csv', str(a)+b+'CARS_spectrums_valid.csv')
    dataset_val = raman_dataset('data', 'CARS1.csv', 'CARS1.csv')

    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    checkpoint_path = os.path.join(model_save_dir, 'checkpoint'+str(args.dataset)+'.pth.tar')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()
    val_loss = utils.AverageMeter() # validation loss
    with torch.no_grad():
        results=[]
        for val_step, inputs in enumerate(tqdm(val_loader)):
            raman = inputs[0].float().cuda()
            cars = inputs[1].float().cuda()
            outputs = model(cars)
            results.append((outputs.cpu()).numpy())
            loss_valid = loss_function(outputs, raman)
            val_loss.update(loss_valid.item(), raman.size(0))
        print(np.size(results))
        results = np.array(results)
        results = results.reshape(results.shape[1],results.shape[2])
        print(np.size(results))
        pd.DataFrame(results).to_csv('./data/'+str(a)+b+'Raman_spectrums_results.csv')
    print('----validation----')
    print_string = 'loss: {loss:.5f}'.format(loss=val_loss.avg)
    print(print_string)
    


