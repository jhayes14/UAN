import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import models
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler

import scipy.spatial
from tqdm import tqdm
from collections import defaultdict
from scipy import misc
import time
import math

from utils import *
import attack_model
from models import *
from pretrained_models_pytorch import pretrainedmodels

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--log_interval', type=int, default=1, help='how many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

parser.add_argument('--optimize_on_success', type=int, default=0, help="whether to optimize on samples that are already successful \
                     - if set to 0, we only optimize on failed attempts to compute adv examples, removing the successes \
                     - if set to 1, we only optimize on all samples \
                     - if set to 2 this tries to build an equilibrium between highest and sec highest classes") 
parser.add_argument('--targeted', type=int, default=0, help='if the attack is targeted (default False)')
parser.add_argument('--chosen_target_class', type=int, default=0, help='int representing class to target')
parser.add_argument('--restrict_to_correct_preds', type=int, default=1, help='if 1, only compute adv examples on correct predictions')
parser.add_argument('--shrink', type=float, default=0.01, help='weight for misclassification success by attacker')
parser.add_argument('--shrink_inc', type=float, default=0.01, help='weight for misclassification success by attacker')
parser.add_argument('--ldist_weight', type=float, default=4.0, help='how much to weight the linf loss term')
parser.add_argument('--l2reg', type=float, default=0.01, help='weight for misclassification success by attacker')
parser.add_argument('--max_norm', type=float, default=0.04, help='max allowed perturbation')
parser.add_argument('--norm', type=str, default='linf', help='l2 or linf')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--every', type=int, default=1, help='save if epoch is divisible by this')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--imageSize', type=int, default=299, help='the height / width of the input image to network')
parser.add_argument('--netAttacker', default='', help="path to netAttacker (to continue training)")
parser.add_argument('--netClassifier', default='./checkpoint/ckpt.t7', help="For CIFAR-10: path to netClassifier (to get target model predictions) \
                                                                             For ImageNet: type of classifier (e.g. inceptionV3)")
parser.add_argument('--outf', default='./logs', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=5198, help='manual seed')
parser.add_argument('--dataset', type=str, default='ImageNet', help='dataset images path')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

WriteToFile('./%s/log' %(opt.outf), opt)


class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor

class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

nc = 3
ngpu = int(opt.ngpu)


# set-up models and load weights if any are saved
netAttacker = attack_model._netAttacker(ngpu, opt.imageSize)
netAttacker.apply(weights_init)
if opt.netAttacker != '':
    netAttacker.load_state_dict(torch.load(opt.netAttacker))

print("=> creating model ")
if opt.dataset == 'cifar10':
    checkpoint = torch.load(opt.netClassifier)
    netClassifier = checkpoint['net']

elif opt.dataset == 'ImageNet':
    netClassifier = pretrainedmodels.__dict__[opt.netClassifier](num_classes=1000, pretrained='imagenet')


if opt.cuda:
    netAttacker.cuda()
    netClassifier.cuda()


print('==> Preparing data..')
if opt.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.Scale((opt.imageSize,opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.Scale((opt.imageSize,opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = dset.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=2)
    testset = dset.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



elif opt.dataset == 'ImageNet':
    normalize = transforms.Normalize(mean=netClassifier.mean,
                                     std=netClassifier.std)
 
    idx = np.arange(50000)
    np.random.shuffle(idx)
    training_idx = idx[:40000]
    test_idx = idx[40000:]
    
    train_loader = torch.utils.data.DataLoader(
        dset.ImageFolder('./imagenet/data/val/', transforms.Compose([
            transforms.Scale(round(max(netClassifier.input_size)*1.050)),
            transforms.CenterCrop(max(netClassifier.input_size)),
            transforms.ToTensor(),
            ToSpaceBGR(netClassifier.input_space=='BGR'),
            ToRange255(max(netClassifier.input_range)==255),
            normalize,
        ])),
        batch_size=opt.batchSize, shuffle=False, sampler=SubsetRandomSampler(training_idx),
        num_workers=opt.workers, pin_memory=True)
 
    test_loader = torch.utils.data.DataLoader(
        dset.ImageFolder('./imagenet/data/val/', transforms.Compose([
            transforms.Scale(round(max(netClassifier.input_size)*1.050)),
            transforms.CenterCrop(max(netClassifier.input_size)),
            transforms.ToTensor(),
            ToSpaceBGR(netClassifier.input_space=='BGR'),
            ToRange255(max(netClassifier.input_range)==255),
            normalize,
        ])),
        batch_size=opt.batchSize, shuffle=False, sampler=SubsetRandomSampler(test_idx),
        num_workers=opt.workers, pin_memory=True)
 
# setup optimizer
optimizerAttacker = optim.Adam(netAttacker.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.l2reg)

# pre-set noise variable
noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
if opt.cuda:
    noise = noise.cuda()
noise = Variable(noise)

def train(epoch, c, noise):
    # set-up structures to track norms, losses etc.
    netAttacker.train()
    netClassifier.eval()
    c_loss, L_inf, L2, pert_norm, dist, adv_norm, non_adv_norm = [ ], [ ], [ ], [ ], [ ], [ ], [ ] 
    total_count, success_count, skipped, no_skipped = 0, 0, 0, 0
     
    for batch_idx, (inputv, cls) in enumerate(train_loader):
 
        optimizerAttacker.zero_grad()
        batch_size = inputv.size(0)
        targets = torch.LongTensor(batch_size)
        if opt.cuda:
            inputv = inputv.cuda()
            targets = targets.cuda()
        inputv = Variable(inputv)
        targets = Variable(targets)

        prediction = netClassifier(inputv)

        # only computer adversarial examples on examples that are originally classified correctly        
        if opt.restrict_to_correct_preds == 1:
            if opt.cuda:
                cls = cls.cuda()
            # get indexes where the original predictions are incorrect
            incorrect_idxs = np.array( np.where(prediction.data.max(1)[1].eq(cls).cpu().numpy() == 0))[0].astype(int)
            skipped += incorrect_idxs.shape[0]
            no_skipped += (batch_size - incorrect_idxs.shape[0])
            if incorrect_idxs.shape[0] == batch_size:
                #print("All original predictions were incorrect! Skipping batch!")
                continue
            elif incorrect_idxs.shape[0] > 0 and incorrect_idxs.shape[0] < batch_size:
                # get indexes of the correct predictions and filter out the incorrect indexes
                correct_idxs = np.setdiff1d( np.arange(batch_size), incorrect_idxs)
                correct_idxs = torch.LongTensor(correct_idxs)
                if opt.cuda:
                    correct_idxs = correct_idxs.cuda()
                inputv = torch.index_select(inputv, 0, Variable(correct_idxs))
                prediction = torch.index_select(prediction, 0, Variable(correct_idxs))
                cls = torch.index_select(cls, 0, correct_idxs)

        # if this is a targeted attack, fill the target variable and filter out examples that are of that target class 
        if opt.targeted == 1:
            targets.data.resize_as_(cls).fill_(opt.chosen_target_class) 
            ids = np.array( np.where(targets.data.eq(cls).cpu().numpy() == 0))[0].astype(int)
            ids = torch.LongTensor(ids)
            if opt.cuda:
                ids = ids.cuda()
            inputv = torch.index_select(inputv, 0, Variable(ids))
            prediction = torch.index_select(prediction, 0, Variable(ids))
            cls = torch.index_select(cls, 0, ids)

        # update sizes
        batch_size = inputv.size(0)
        noise.data.resize_(batch_size, opt.nz, 1, 1).normal_(0, 0.5)
        targets.data.resize_(batch_size)
       
        # compute an adversarial example and its prediction 
        prediction = netClassifier(inputv)
        delta = netAttacker(noise)
        adv_sample_ = delta*c + inputv
        adv_sample = torch.clamp(adv_sample_, min_val, max_val) 
        adv_prediction = netClassifier(adv_sample)
        
        # get indexes of failed adversarial examples
        if opt.targeted == 1:
            no_idx = np.array( np.where(adv_prediction.data.max(1)[1].eq(targets.data).cpu().numpy() == 0))[0].astype(int)
        else:
            no_idx = np.array( np.where(adv_prediction.data.max(1)[1].eq(prediction.data.max(1)[1]).cpu().numpy() == 1))[0].astype(int)

        # update success and total counts         
        success_count += inputv.size(0) - len(no_idx)
        total_count += inputv.size(0)     

        # if there are any adversarial examples, compute distance and update norms, and save image
        if len(no_idx) != inputv.size(0):
            yes_idx = np.setdiff1d(np.array(range(inputv.size(0))), no_idx)
            for i, adv_idx in enumerate(yes_idx):
                clean = inputv[adv_idx].data.view(1, nc, opt.imageSize ,opt.imageSize)
                adv = adv_sample[adv_idx].data.view(1, nc, opt.imageSize, opt.imageSize)
                pert = (inputv[adv_idx]-adv_sample[adv_idx]).data.view(1, nc, opt.imageSize, opt.imageSize)  
               
                if opt.dataset == 'cifar10': 
                    adv_ = rescale(adv_sample[adv_idx], mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                    clean_ = rescale(inputv[adv_idx], mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                elif opt.dataset == 'ImageNet': 
                    adv_ = rescale(adv_sample[adv_idx], mean=netClassifier.mean, std=netClassifier.std)
                    clean_ = rescale(inputv[adv_idx], mean=netClassifier.mean, std=netClassifier.std)
                
                linf = torch.max(torch.abs(adv_ - clean_)).data.cpu().numpy()
                noise_norm = torch.sqrt(torch.sum( (clean_[:, :, :] - adv_[:, :, :])**2  )).data.cpu().numpy()
                image_norm = torch.sqrt(torch.sum( clean_[:, :, :]**2 )).data.cpu().numpy()
                adv_norm_s   = torch.sqrt(torch.sum( adv_[:, :, :]**2 )).data.cpu().numpy()
                
                dist.append(noise_norm/image_norm)
                pert_norm.append(noise_norm)
                non_adv_norm.append(image_norm)
                adv_norm.append(adv_norm_s)
                L_inf.append(linf)

                if batch_idx == 0: 
                    vutils.save_image(torch.cat((clean,pert,adv)), './{}/{}_{}.png'.format(opt.outf, epoch, i), normalize=True, scale_each=True)
   
        # if opt.optimize_on_success == 0, we do not optimize on already successfully computed adversarial examples
        # we remove them from consideration 
        if opt.optimize_on_success == 0:
            if len(no_idx)!=0:  
                # select the non adv examples to optimise on 
                no_idx = torch.LongTensor(no_idx)
                if opt.cuda:
                    no_idx = no_idx.cuda()
                no_idx = Variable(no_idx)
                inputv = torch.index_select(inputv, 0, no_idx)
                prediction = torch.index_select(prediction, 0, no_idx)
                targets = torch.index_select(targets, 0, no_idx)
                adv_prediction = torch.index_select(adv_prediction, 0, no_idx)
                delta = torch.index_select(delta, 0, no_idx)
                adv_sample = torch.index_select(adv_sample, 0, no_idx)

        # if opt.optimize_on_success == 1, we continue to optimize on already successfully computed adversarial examples
        # by maximizing the distance between the adversarially predicted class and the target class 
        elif opt.optimize_on_success == 1:
            yes_idx = np.setdiff1d(np.arange(batch_size), no_idx)
            if yes_idx.shape[0]!=0:
                adv_prediction_succ = adv_prediction[torch.LongTensor(yes_idx).cuda()]
                prediction_succ = prediction[torch.LongTensor(yes_idx).cuda()].data.max(1)[1]
                adv_prediction = adv_prediction[torch.LongTensor(no_idx).cuda()]
                adv_pred_idx = torch.FloatTensor([x[prediction_succ[i]].data[0] for i, x in enumerate(adv_prediction_succ)]).cuda()
                adv_max_idx = adv_prediction_succ.data.max(1)[0]
                success_loss = -torch.mean( torch.log(adv_max_idx)-torch.log(adv_pred_idx) )
            else:
                success_loss = 0


        # if opt.optimize_on_success == 2, we do nothing, causing the predicted and target (second max in untargeted) class to oscillate as the max
        elif opt.optimize_on_success == 2:
            pass

        # compute loss and backprop
        adv_prediction_softmax = F.softmax(adv_prediction)
        #adv_prediction_np = adv_prediction.data.cpu().numpy()
        adv_prediction_np = adv_prediction_softmax.data.cpu().numpy()
        curr_adv_label = Variable(torch.LongTensor( np.array( [arr.argsort()[-1] for arr in adv_prediction_np] ) ) )
        if opt.targeted == 1:
            targ_adv_label = Variable(torch.LongTensor( np.array( [targets.data[i] for i, arr in enumerate(adv_prediction_np)] ) ) )
        else:
            targ_adv_label = Variable(torch.LongTensor( np.array( [arr.argsort()[-2] for arr in adv_prediction_np] ) ) )
        if opt.cuda:
            curr_adv_label = curr_adv_label.cuda()
            targ_adv_label = targ_adv_label.cuda()
        curr_adv_pred = adv_prediction_softmax.gather(1, curr_adv_label.unsqueeze(1))
        targ_adv_pred = adv_prediction_softmax.gather(1, targ_adv_label.unsqueeze(1))
        if opt.optimize_on_success == 1:
             classifier_loss = torch.mean( torch.log(curr_adv_pred)-torch.log(targ_adv_pred) ) + success_loss
        else:
             classifier_loss = torch.mean( torch.log(curr_adv_pred)-torch.log(targ_adv_pred) )

        if opt.norm == 'linf':
            ldist = opt.ldist_weight*torch.max(torch.abs(adv_sample - inputv))
        elif opt.norm == 'l2':
            ldist = opt.ldist_weight*torch.mean(torch.sqrt(torch.sum( (adv_sample - inputv)**2  )))
        else:
            print("Please define a norm (l2 or linf)")
            exit()
        loss = classifier_loss + ldist_loss 
        loss.backward()
        optimizerAttacker.step()
        c_loss.append(classifier_loss.data[0])
        
        # log to file  
        progress_bar(batch_idx, len(train_loader), "Tr E%s, C_L %.5f A_Succ %.5f L_inf %.5f L2 %.5f (Pert %.2f, Adv %.2f, Clean %.2f) C %.6f Skipped %.1f%%" %(epoch, np.mean(c_loss), success_count/total_count, np.mean(L_inf), np.mean(dist), np.mean(pert_norm), np.mean(adv_norm), np.mean(non_adv_norm), c, 100*(skipped/(skipped+no_skipped)))) 
        WriteToFile('./%s/log' %(opt.outf),  "Tr Epoch %s batch_idx %s C_L %.5f A_Succ %.5f L_inf %.5f L2 %.5f (Pert %.2f, Adv %.2f, Clean %.2f) C %.6f Skipped %.1f%%" %(epoch, batch_idx, np.mean(c_loss), success_count/total_count, np.mean(L_inf), np.mean(dist), np.mean(pert_norm), np.mean(adv_norm), np.mean(non_adv_norm), c, 100*(skipped/(skipped+no_skipped))))

    # save model weights 
    if epoch % opt.every == 0:
        torch.save(netAttacker.state_dict(), '%s/netAttacker_%s.pth' % (opt.outf, epoch))

    return success_count/total_count, np.mean(L_inf), np.mean(dist)


def test(epoch, c, noise):
    netAttacker.eval()
    netClassifier.eval()
    L_inf = [ ]
    L2 = [ ]
    pert_norm = [ ]
    dist = [ ]
    adv_norm = [ ]
    non_adv_norm = [ ]
    total_count = 0
    success_count = 0
    skipped = 0
    no_skipped = 0
    for batch_idx, (inputv, cls) in enumerate(test_loader):
        if opt.cuda:
            inputv = inputv.cuda()
        inputv = Variable(inputv)
        batch_size = inputv.size(0)
 
        targets = torch.LongTensor(batch_size)
        if opt.cuda:
            targets = targets.cuda()
        targets = Variable(targets)
        
        prediction = netClassifier(inputv)
        
        if opt.restrict_to_correct_preds == 1:
            if opt.cuda:
                cls = cls.cuda()
            incorrect_idxs = np.array( np.where(prediction.data.max(1)[1].eq(cls).cpu().numpy() == 0))[0].astype(int)
            skipped += incorrect_idxs.shape[0]
            no_skipped += (batch_size - incorrect_idxs.shape[0])
            if incorrect_idxs.shape[0] == batch_size:
                print("All original predictions were incorrect! Skipping batch!")
                continue
            elif incorrect_idxs.shape[0] > 0 and incorrect_idxs.shape[0] < batch_size:
                correct_idxs = np.setdiff1d( np.arange(batch_size), incorrect_idxs)
                correct_idxs = torch.LongTensor(correct_idxs)
                if opt.cuda:
                    correct_idxs = correct_idxs.cuda()
                inputv = torch.index_select(inputv, 0, Variable(correct_idxs))
                prediction = torch.index_select(prediction, 0, Variable(correct_idxs))
                cls = torch.index_select(cls, 0, correct_idxs)
        # remove samples that are of the target class
        if opt.targeted == 1:
            targets.data.resize_as_(cls).fill_(opt.chosen_target_class) 
            ids = np.array( np.where(targets.data.eq(cls).cpu().numpy() == 0))[0].astype(int)
            ids = torch.LongTensor(ids)
            if opt.cuda:
                ids = ids.cuda()
            inputv = torch.index_select(inputv, 0, Variable(ids))
            prediction = torch.index_select(prediction, 0, Variable(ids))
            cls = torch.index_select(cls, 0, ids)

        batch_size = inputv.size(0)
        noise.data.resize_(batch_size, opt.nz, 1, 1).normal_(0, 0.5)
        targets.data.resize_(batch_size)
        
        prediction = netClassifier(inputv)
        delta = netAttacker(noise)
        adv_sample_ = delta*c + inputv
        adv_sample = torch.clamp(adv_sample_, min_val, max_val) 
        adv_prediction = netClassifier(adv_sample)
        
        if opt.targeted == 1:
            no_idx = np.array( np.where(adv_prediction.data.max(1)[1].eq(targets.data).cpu().numpy() == 0))[0].astype(int)
        else:
            no_idx = np.array( np.where(adv_prediction.data.max(1)[1].eq(prediction.data.max(1)[1]).cpu().numpy() == 1))[0].astype(int)
  
        success_count += inputv.size(0) - len(no_idx)
        total_count += inputv.size(0)     
        if len(no_idx) != inputv.size(0):
            yes_idx = np.setdiff1d(np.array(range(inputv.size(0))), no_idx)
            for i, adv_idx in enumerate(yes_idx):
                clean = inputv[adv_idx].data.view(1, nc, opt.imageSize ,opt.imageSize)
                adv = adv_sample[adv_idx].data.view(1, nc, opt.imageSize, opt.imageSize)
                pert = (inputv[adv_idx]-adv_sample[adv_idx]).data.view(1, nc, opt.imageSize, opt.imageSize)  
                
                if opt.dataset == 'cifar10': 
                    adv_ = rescale(adv_sample[adv_idx], mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                    clean_ = rescale(inputv[adv_idx], mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                elif opt.dataset == 'ImageNet': 
                    adv_ = rescale(adv_sample[adv_idx], mean=netClassifier.mean, std=netClassifier.std)
                    clean_ = rescale(inputv[adv_idx], mean=netClassifier.mean, std=netClassifier.std)

                linf = torch.max(torch.abs(adv_ - clean_)).data.cpu().numpy()
                noise_norm = torch.sqrt(torch.sum( (clean_[:, :, :] - adv_[:, :, :])**2  )).data.cpu().numpy()
                image_norm = torch.sqrt(torch.sum( clean_[:, :, :]**2 )).data.cpu().numpy()
                adv_norm_s   = torch.sqrt(torch.sum( adv_[:, :, :]**2 )).data.cpu().numpy()
                
                dist.append(noise_norm/image_norm)
                pert_norm.append(noise_norm)
                non_adv_norm.append(image_norm)
                adv_norm.append(adv_norm_s)
                L_inf.append(linf)
                
        progress_bar(batch_idx, len(test_loader), "Val E%s, A_Succ %.5f L_inf %.5f L2 %.5f (Pert %.2f, Adv %.2f, Clean %.2f) C %.6f Skipped %.1f%%" %(epoch, success_count/total_count, np.mean(L_inf), np.mean(dist), np.mean(pert_norm), np.mean(adv_norm), np.mean(non_adv_norm), c, 100*(skipped/(skipped+no_skipped)))) 
        WriteToFile('./%s/log' %(opt.outf),  "Val Epoch %s batch_idx %s A_Succ %.5f L_inf %.5f L2 %.5f (Pert %.2f, Adv %.2f, Clean %.2f) C %.6f Skipped %.1f%%" %(epoch, batch_idx, success_count/total_count, np.mean(L_inf), np.mean(dist), np.mean(pert_norm), np.mean(adv_norm), np.mean(non_adv_norm), c, 100*(skipped/(skipped+no_skipped))))

if __name__ == '__main__':
    
    c = opt.shrink
    if opt.dataset == 'cifar10':
        min_val, max_val = find_boundaries(train_loader)
    elif opt.dataset == 'ImageNet':
        mins = np.array([netClassifier.input_range[0]]*3)
        maxs = np.array([netClassifier.input_range[1]]*3)
        min_val = np.min((mins-np.array(netClassifier.mean))/np.array(netClassifier.std))
        max_val = np.max((maxs-np.array(netClassifier.mean))/np.array(netClassifier.std))
    print(min_val, max_val)
    for epoch in range(1, opt.epochs + 1):
        start = time.time()
        score, linf, l2 = train(epoch, c, noise)
        if linf > opt.max_norm:
            break
        if l2 > opt.max_norm:
            break
        end = time.time()
        if score >= 1.00:
            break
        if epoch == 1:
            curr_pred = score
        if epoch % 2 == 0:
            prev_pred = curr_pred
            curr_pred = score
        if epoch > 2:
            if ( prev_pred - curr_pred ) >= 0:
                c += opt.shrink_inc
    test(epoch, c, noise)
