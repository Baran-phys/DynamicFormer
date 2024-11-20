import os
import sys
import gc
import random
import numpy as np
import pandas as pd
import ast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Sampler, Dataset

from tqdm import tqdm, trange


TOKENS = {
    '<eos>': 0
}

class TRDataset(Dataset):
    """TR amplitude, F_gn, Dataset."""

    def __init__(self, csv_file, log_scale=True, mode='train'):
        """Initializes instance of class TRDataset.

        Args:
            csv_file (str): Path to the csv file with the amplitudes data.

        """
        self.log_scale = log_scale

        chunk = pd.read_csv(csv_file, chunksize=64)
        df = pd.concat(chunk)
        if mode == 'train':
            df = df[df['g']<=13]
        else:
            df = df[df['g']>=14]
            
         
        df = df.applymap(lambda x: np.array(ast.literal_eval(str(x).replace('{}', '{{{0.}}}').replace('{', '[').replace('}', ']').replace('*^', 'e')),
                                            dtype=float))
        
        df['Permutations'] = df.apply(lambda row: np.delete(row['Permutations'], 
                                                            np.where(row['Fgn'] == 0)[0], axis=0), axis = 1)
        df['Fgn'] = df.apply(lambda row: row['Fgn'][row['Fgn'] != 0], axis = 1)
        df = df.explode(['Permutations','Fgn'], ignore_index=True)
        df[['Fgn']] = df[['Fgn']].apply(pd.to_numeric)
        df.drop(df[df['g']==1][df['n']==2].index, inplace=True)
        df.drop(df[df['g']==1][df['n']==1].index, inplace=True)
        df.drop(df[df['g']==0][df['n']==3].index, inplace=True)
        df.drop(df[df['g']==0][df['n']==4].index, inplace=True)
        df.drop(df[df['g']==0][df['n']==5].index, inplace=True)


        self.g = df.iloc[:,0].to_numpy()#.values.tolist()
        self.n = df.iloc[:,1].to_numpy()#.values.tolist()
        self.b = df.iloc[:,3].to_numpy()#.values.tolist()
        self.c = df.iloc[:,4].to_numpy()#.values.tolist()
        self.x = df.iloc[:,6].to_numpy()#.values.tolist() #x=df.iloc[:,0:7].values.tolist()
        self.y = df.iloc[:,7].to_numpy()#.values.tolist()

        del [[df]]
        gc.collect()
        df=pd.DataFrame()


    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx):
        X = torch.tensor(self.x[idx], dtype=torch.int8) #max value 50: dict size
        card = X.shape[0]
        xpd = (0, 52-card) #22 --> 52
        X_padded = F.pad(X, xpd, "constant", 0)
        permutations = X_padded#.int8 
        
        mask_x = torch.zeros(52, dtype=torch.bool)
        mask_x[:card] = 1
        
        #mask_x = torch.ones(52, dtype=torch.bool)
        #bool_x = torch.zeros(card, dtype=torch.bool)
        #mask_x[:card] = bool_x[:]

        G = torch.tensor(self.g[idx], dtype=torch.int8)
        N = torch.tensor(self.n[idx], dtype=torch.int8)
        gn = torch.stack([G, N])#.int8
        
        #Creating coo of C tensor, pad it, and generating the mask
        cz = np.zeros((np.shape(np.nonzero(self.c[idx]))[1], 4)) #xyz = np.zeros((400, 4))
        cz[:, 0] = np.nonzero(self.c[idx])[0]
        cz[:, 1] = np.nonzero(self.c[idx])[1]
        cz[:, 2] = np.nonzero(self.c[idx])[2]
        cz[:, -1] = np.exp(self.c[idx][self.c[idx]!=0])
        
        mask_c = np.full((1500, 4), True, dtype=bool)
        bool_c = np.full((cz.shape), False, dtype=bool)
        mask_c[:cz.shape[0]] = bool_c[:]
        mask_c = torch.tensor(mask_c.all(-1))
        
        cpd = (0, 0, 0, 1500-np.shape(cz)[0]) #max:378(400) --> 1275(1500)
        cz = F.pad(torch.tensor(cz), cpd, "constant", 0)
        
        #Creating coo of B tensor, pad it, and generating the mask
        bz = np.zeros((np.shape(np.nonzero(self.b[idx]))[1], 4))
        #bz = np.zeros((np.shape(np.nonzero(self.b[idx]))[1], 1))
        bz[:, 0] = np.nonzero(self.b[idx])[0]
        bz[:, 1] = np.nonzero(self.b[idx])[1]
        bz[:, 2] = np.nonzero(self.b[idx])[2]
        #bz[:, -1] = np.log(self.b[idx][self.b[idx]!=0])
        bz[:, -1] = np.log(1 +100*self.b[idx][self.b[idx]!=0])
        #bz[:, -1] = np.log1p(self.b[idx][self.b[idx]!=0])
        #bz[:, -1] = self.b[idx][self.b[idx]!=0]
        
        mask_b = np.full((1500, 4), True, dtype=bool)
        #mask_b = np.full((1500, 1), True, dtype=bool)
        bool_b = np.full((bz.shape), False, dtype=bool)
        mask_b[:bz.shape[0]] = bool_b[:]
        mask_b = torch.tensor(mask_b.all(-1))
        
        bpd = (0, 0, 0, 1500-np.shape(bz)[0])  #max:462(470) --> 1428(1500)
        bz = F.pad(torch.tensor(bz, dtype=torch.float32), bpd, "constant", 0)
        
        amplitudes = torch.tensor(self.y[idx], dtype=torch.float32)
        if self.log_scale:
            amplitudes = torch.log(amplitudes)#.double() #.unsqueeze(1)
            
        return permutations, amplitudes, gn, bz, cz, mask_x, mask_b, mask_c