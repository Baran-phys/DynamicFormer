#!/opt/anaconda3/bin/python
# coding: utf-8


import os
import sys
import math
import random
import shutil
import json
from datetime import datetime
from pprint import pprint
from typing import Tuple, Union, Optional
import inspect
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm, trange
from sklearn.metrics import mean_squared_error,  mean_absolute_error, r2_score
import ast
import copy
import wandb


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Sampler, Dataset
import torch.utils.data as data_utils
from torchmetrics import MeanSquaredError, MeanAbsoluteError, SpearmanCorrCoef, Accuracy

from x_transformers import TransformerWrapper, ContinuousTransformerWrapper, Decoder, Encoder, CrossAttender

import layers
from layers import PNAAggregator, LePNAAggregator
import TR_data

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'



# ====================================================
# CFG
# ====================================================

#Last Note, TR_15: #notes = "run 8, new data, masking x,b,c, lr 0.001"
class CFG:
    run_name = "TR"
    epochs = 100
    lr = 0.0005
    wd = 0.00001 #0.00001
    step_size = 14 
    gamma = 0.5 
    bs = 64
    train_val_ratio = 80
    res_dir = './results'
    method = 'LePNA'
    log_scale = True
    schuffle = False
    save = True
    gpu = 2
    


def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


#Model


class DRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1), requires_grad=True)
        self.b = nn.Parameter(torch.randn(1), requires_grad=True)
        self.c = nn.Parameter(torch.randn(1), requires_grad=True)
        self.d = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        term1 = x + (self.a * torch.square(torch.sin(self.a * x)) / self.b)
        term2 = self.c * torch.cos(self.b * x)
        term3 = self.d * torch.tanh(self.b * x)
        return term1 + term2 + term3



class ampTR(nn.Module):
    def __init__(self, in_features, out_features, hidden_mlp, cfg=None):
        super(ampTR, self).__init__()
        self.gn_mlp = layers.MLP_dra(2, 64, 128, 1, BN=True) #127

        self.b_embedding = layers.MLP_dra(4, 64, 128, 2, dim_in_2=64, modulation="+")
        #self.c_embedding = layers.MLP_dra(4, 64, 128, 2, dim_in_2=64, modulation="+")

        self.perm_emb = nn.Embedding(55, 64) #29+2 --> 55(52)
        self.perm_embedding = layers.MLP_dra(64, 64, 128, 2, dim_in_2=64, modulation="+")
        
        self.nonlin_b = DRA()
        #self.nonlin_c = DRA()
        self.nonlin_x = DRA()
        
        
        self.trans_1 = ContinuousTransformerWrapper(
            dim_in = 64,
            dim_out = 64,
            max_seq_len = 1500,
            emb_dropout = 0.1,
            use_abs_pos_emb = False,
            num_memory_tokens = 1,
            attn_layers = Encoder(
                dim = 256,
                depth = 1,
                heads = 4,
                dynamic_pos_bias = True,
                dynamic_pos_bias_log_distance = False,
                use_rmsnorm = True,
                attn_dropout = 0.1,
                layer_dropout = 0.1,
                ff_dropout = 0.1,
                ff_dra = True,
                ff_no_bias = True,
                attn_qk_norm = True, 
                attn_qk_norm_groups = 8,    
            )
        )
                
        self.trans_2 = ContinuousTransformerWrapper(
            dim_in = 64,
            dim_out = 64,
            max_seq_len = 55,
            emb_dropout = 0.1,
            use_abs_pos_emb = False,
            num_memory_tokens = 1,
            attn_layers = Encoder(
                dim = 256,
                depth = 1,
                heads = 4,
                use_rmsnorm = True,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                layer_dropout = 0.1,
                ff_dra = True,
                ff_no_bias = True,
                attn_qk_norm = True, 
                attn_qk_norm_groups = 8,  
            )
        )
        
        

        self.pooling_x = LePNAAggregator(average_n=9)
        self.pooling_bc = LePNAAggregator(average_n=9)
        self.nonlin = DRA()
        self.last_norm = nn.BatchNorm1d(832, affine=True)

        self.bn1 = nn.BatchNorm1d(256, affine=False)
        self.bn2 = nn.BatchNorm1d(256, affine=False)

        #self.final_mlp = layers.MLP_dra(832, 1, 128, 1, BN = False)
        self.final_mlp = layers.MLP(832, 1, 128, 1)
        
    def off_diagonal(self, x):
        n, m = x.shape
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    
    def calc_wasserstein_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        z = torch.cat((z1, z2), 0)
        N = z.size(0)
        D = z.size(1)
    
        z_center = torch.mean(z, dim=0, keepdim=True)
        mean = z.mean(0)
        covariance = torch.mm((z - z_center).t(), (z - z_center)) / N + 1e-12 * torch.eye(D).cuda()
    
        # calculation of part1
        part1 = torch.sum(torch.multiply(mean, mean))
    
        # Use torch.linalg.eig to compute the eigenvalues and eigenvectors of the covariance matrix
        L_complex, V_complex = torch.linalg.eig(covariance)
        S = torch.abs(L_complex.real)  # Take the real part and absolute value for eigenvalues
    
        # Diagonal matrix of the square roots of eigenvalues
        mS = torch.sqrt(torch.diag(S))
    
        # Convert mS to complex type to match V_complex
        mS_complex = mS.to(dtype=V_complex.dtype)
    
        # Construct the modified covariance matrix using the eigenvectors and mS
        covariance2 = torch.mm(torch.mm(V_complex, mS_complex), V_complex.T)
    
        # calculation of part2
        part2 = torch.trace(covariance - 2.0 / math.sqrt(D) * covariance2.real)  # Take the real part if needed
        wasserstein_loss = torch.sqrt(part1 + 1 + part2)
    
        return wasserstein_loss


    def forward(self, x, z, b, c, mask_x, mask_b, mask_c):
        gn = self.gn_mlp(z)
        x = self.perm_emb(x)
        
        x = self.nonlin_x(self.perm_embedding(x, gn.unsqueeze(1)))
        b = self.nonlin_b(self.b_embedding(b, gn.unsqueeze(1), mask = ~mask_b))
        #c = self.nonlin_c(self.c_embedding(c, gn.unsqueeze(1), mask = ~mask_c))

        bemb, inter_b = self.trans_1(b, mask = ~mask_b, return_intermediates = True)
        bemb = bemb + b
        #cemb = self.trans_1(c, mask = ~mask_c) # BxNx(64)
        #cemb = cemb + c
        perm, inter_x = self.trans_2(x, mask = mask_x, return_intermediates = True)
        perm = perm + x

        
        x = self.pooling_x(perm)
        b = self.pooling_bc(bemb)
        #c = self.pooling_bc(cemb)
        x = self.nonlin(b + x)
        x = torch.cat([x, gn], dim=1) #[bs, 416] or 832
        x = self.last_norm(x)
        
        out = self.final_mlp(x)

        inter_x = inter_x.memory_tokens.squeeze(1)
        inter_b = inter_b.memory_tokens.squeeze(1)
        bz = inter_x.shape[0]
        cor = self.bn1(inter_x).T @ self.bn2(inter_b)
        cor.div_(bz)
        on_diag = torch.diagonal(cor).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(cor).pow_(2).sum()
        bt = on_diag + 0.5 * off_diag
        
        #uniformity = self.calc_wasserstein_loss(inter_x, inter_b)
        
        return out, bt#, uniformity #[bs, 1]




def update_info(loss, pred, amps, accum_info):
    accum_info['loss'] += loss.item()

    pred = pred.detach().cpu().numpy()
    amps = amps.detach().cpu().numpy()

    # Compute exponential in numpy
    #pred_amps = np.exp(pred).astype(np.float64)
    #amps = np.exp(amps).astype(np.float64)
    pred_amps = pred.astype(np.float32)
    amps = amps.astype(np.float32)

    # Calculate MSE and MAE using numpy/scikit-learn
    mse_value = mean_squared_error(amps, pred_amps, multioutput='raw_values')
    mae_value = mean_absolute_error(amps, pred_amps, multioutput='raw_values')

    accum_info['mse'] += mse_value
    accum_info['mae'] += mae_value

    return accum_info


#def uniform_loss(x, t=2):
#    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


#def log_double_factorial(n):
#    """Compute the logarithm of the double factorial using gamma function for non-integer values."""
#    result = torch.zeros_like(n, dtype=torch.float32).to(n.device)
#    even_idx = (n % 2 == 0)
#    odd_idx = ~even_idx
#
#    result[even_idx] = (n[even_idx] // 2) * torch.log(torch.tensor(2.0)) + torch.lgamma(n[even_idx] // 2 + 1)
#    result[odd_idx] = (n[odd_idx] / 2) * torch.log(torch.tensor(2.0)) + torch.lgamma(n[odd_idx] / 2 + 1) + torch.log(torch.tensor(math.sqrt(2 / math.pi)))
#
#    return result
#
#def intersection_number_asymptotic(g, n, ds):
#    """Compute the asymptotic intersection number given genus g and degrees ds using log computations."""
#    log_top = log_double_factorial(6 * g - 5 + 2 * n)
#    log_bottom_genus = g * torch.log(torch.tensor(24.0)) + torch.lgamma(g + 1)
#    log_bottom_degrees = torch.sum(log_double_factorial(2 * ds + 1), dim=1)
#    
#    log_result = log_top - log_bottom_genus - log_bottom_degrees
#    return log_result#.exp()
#
#def compute_gradient_wrt_g(g, n, ds, epsilon=0.001):
#    """Compute the numerical gradient of the intersection number with respect to g using central differences."""
#    f_g_minus_epsilon = intersection_number_asymptotic(g - epsilon, n, ds)
#    f_g_plus_epsilon = intersection_number_asymptotic(g + epsilon, n, ds)
#    gradient_g = (f_g_plus_epsilon - f_g_minus_epsilon) / (2 * epsilon)
#    return gradient_g



def train_epoch(data, epoch, model, optimizer, device):
    model.train()

    # Iterate over batches
    accum_info = {k: 0.0 for k in ['loss', 'mse', 'mae']}
    for i, (perms, amps, gns, b, c, mask_x, mask_b, mask_c) in enumerate(tqdm(data)):
        optimizer.zero_grad()
        # One Train step on the current batch
        batch_size = perms.shape[0]
        
        perms = perms.int().to(device) #.permute(0,2,1)
        amps = amps.to(device, torch.float32)
        gns = gns.float().to(device)
        gns.requires_grad_(True)
        b = b.float().to(device)
        c = c.float().to(device)
        mask_x = mask_x.to(device)
        mask_b = mask_b.to(device)
        mask_c = mask_c.to(device)

        pred, ssl = model(perms, gns, b, c, mask_x, mask_b, mask_c)

        # calc loss
        mae_loss = nn.L1Loss(reduction = 'sum')
        
        # Calculate the gradients of pred with respect to gns[:, 0]
        #grad_pred = torch.autograd.grad(outputs=pred, 
        #                                inputs=gns, 
        #                                grad_outputs=torch.ones_like(pred),
        #                                retain_graph=True, 
        #                                create_graph=True)[0]
        #grad_pred_gns0 = grad_pred[:, 0]
        #gradient_g = compute_gradient_wrt_g(gns[:,0], gns[:,1], perms.float()).detach()
        ## Calculate L1 loss between pred and gradients
        #grad_l1_loss = mae_loss(grad_pred_gns0, gradient_g)
        
        #loss = mae_loss(pred.squeeze(), amps)
        ssl_loss = ssl #+ 0.5 * unifo
        #ssl_loss = uniform_loss(inter_x) + uniform_loss(inter_b)
        loss_mae = mae_loss(pred.squeeze(), amps)
        loss = loss_mae + 0.1 * ssl_loss #+ 0.01 * grad_l1_loss
            
        if i % 500 == 0:
            #print("Train itr %d loss %f" % (i, (loss/batch_size).item()), flush=True)
            print("Train itr %d loss %f ssl_loss %f" % (i, (loss_mae/batch_size).item(), (ssl_loss).item()), flush=True)


        
            

        with torch.no_grad():
            accum_info = update_info(loss, pred.squeeze(), amps, accum_info)

        if torch.isnan(loss):
            raise ValueError("Nan detected in the loss")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

    data_len = len(data.dataset)
    accum_info['loss'] /= data_len
    accum_info['mse'] /= data_len
    accum_info['mae'] /= data_len
    print("train epoch %d loss %f mae %f mse %f" % (epoch, accum_info['loss'], 
                                                    accum_info['mae'],
                                                    accum_info['mse']), flush=True)

    return accum_info




def evaluate(data, epoch, model, device):
    # train epoch
    model.eval()
    accum_info = {k: 0.0 for k in ['loss', 'mse', 'mae']}

    for i, (perms, amps, gns, b, c, mask_x, mask_b, mask_c) in enumerate(tqdm(data)):
        # One Train step on the current batch
        batch_size = perms.shape[0]
        
        perms = perms.int().to(device)#.permute(0,2,1)
        amps = amps.to(device, torch.float32)
        gns = gns.float().to(device)
        c = c.float().to(device)
        b = b.float().to(device)
        mask_x = mask_x.to(device)
        mask_b = mask_b.to(device)
        mask_c = mask_c.to(device)

        pred, _, = model(perms, gns, b, c, mask_x, mask_b, mask_c) # shape (B,N)

        mae_loss = nn.L1Loss(reduction = 'sum')
        loss = mae_loss(pred.squeeze(), amps)
        
        if i % 500 == 0:
            print("Test itr %d loss %f" % (i, (loss/batch_size).item()), flush=True)

        # calc acc, precision, recall
        accum_info = update_info(loss, pred.squeeze(), amps, accum_info)

    data_len = data.dataset.__len__()
    accum_info['loss'] /= data_len
    accum_info['mse'] /= data_len
    accum_info['mae'] /= data_len
    print("validation epoch %d loss %f mae %f mse %f" % (epoch, accum_info['loss'], 
                                                         accum_info['mae'], 
                                                         accum_info['mse']), flush=True)

    return accum_info




def plot_val(df, output_dir, val):
    df.index.name = 'epochs'
    df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    df[['train_'+val, 'val_'+val]].plot(title=val, grid=True, logy=True)
    plt.savefig(os.path.join(output_dir, val+".pdf"))




run = wandb.init(project="AI4TR", 
                 entity="AI4Science",
                 name= CFG.run_name,
                 config=class2dict(CFG),
                 job_type="train")




start_time = datetime.now()

if not os.path.exists(CFG.res_dir):
    os.makedirs(CFG.res_dir)
exp_dir = f'{CFG.run_name}_{start_time:%m%d_%H:%M}'
output_dir = os.path.join(CFG.res_dir, exp_dir)
os.makedirs(output_dir)
print(f'Saving all to {output_dir}')


torch.cuda.set_device(int(CFG.gpu))
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device used:', device)
seed = 1728
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.set_num_threads(4)


print('Training data...', flush=True)
DS_train = TR_data.TRDataset('data_decimals_chi35.csv', log_scale=True, mode='train')
train_loader = DataLoader(DS_train, batch_size = 64, shuffle = True, num_workers = 4, drop_last= True)#, device=device)
print(f'Training data length: {len(train_loader.dataset)}', flush=True)

print('Validation data...', flush=True)
DS_test = TR_data.TRDataset('data_decimals_chi35.csv', log_scale=True, mode='test')
test_loader = DataLoader(DS_test, batch_size = 64, shuffle = True, num_workers = 4, drop_last= True)#, device=device)
print(f'Validation data length: {test_loader.dataset.__len__()}', flush=True)


# Create model instance

model = ampTR(in_features=64, out_features=1, hidden_mlp=256).float()
    
model = model.to(device)
print(f'Model: {model}')
print(f'Num of params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')


optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG.lr, weight_decay = CFG.wd)
# Scheduler setup
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CFG.step_size, gamma=CFG.gamma)

# Metrics
train_loss = np.empty(CFG.epochs, float)
train_mse = np.empty(CFG.epochs, float)
train_mae = np.empty(CFG.epochs, float)

val_loss = np.empty(CFG.epochs, float)
val_mse = np.empty(CFG.epochs, float)
val_mae = np.empty(CFG.epochs, float)

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 100

# Training and evaluation process
for epoch in range(1, CFG.epochs + 1):
    train_info = train_epoch(train_loader, epoch, model, optimizer, device)
    train_loss[epoch - 1], train_mse[epoch - 1], train_mae[epoch - 1] = train_info['loss'], train_info['mse'], train_info['mae']
    
    torch.save(model.state_dict(), os.path.join(output_dir, f'model_e_{epoch}.pth'))
    
    wandb.log({"training_loss":train_info['loss'], 
               "training_mse":train_info['mse'],
               "training_mae":train_info['mae']})

    
    with torch.no_grad():
        val_info = evaluate(test_loader, epoch, model, device)
        wandb.log({"val_loss":val_info['loss'], 
                   "val_mse":val_info['mse'],
                   "val_mae":val_info['mae']})
        
        # deep copy the model
        if val_info['loss'] < best_loss:
            best_loss = val_info['loss']
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(output_dir, 'model_best.pth'))
            
        val_loss[epoch - 1], val_mse[epoch - 1], val_mae[epoch - 1] = val_info['loss'], val_info['mse'], val_info['mae']
    scheduler.step()



# Saving to disk
if CFG.save:
    #shutil.copyfile(os.path.join(output_dir, 'code.py'))
    results_dict = {'train_loss': train_loss,
                    'train_mse': train_mse,
                    'train_mae': train_mae,
                    'val_loss': val_loss,
                    'val_mse': val_mse,
                    'val_mae': val_mae}
    df = pd.DataFrame(results_dict)
    plot_val(df, output_dir, 'loss')
    plot_val(df, output_dir, 'mse')
    plot_val(df, output_dir, 'mae')


print(f'Total runtime: {str(datetime.now() - start_time).split(".")[0]}')
wandb.finish()
torch.cuda.empty_cache()

