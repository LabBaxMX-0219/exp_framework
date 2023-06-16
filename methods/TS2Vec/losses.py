import torch
from torch import nn
import torch.nn.functional as F

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    '''z: (n_instance, n_timestamp) 
        每个timestamp scale都计算loss，dim1==1'''
    while z1.size(1) > 1:

        '''维度大时只计算instance，当timestamp缩减到一定程度后再计算temporal'''
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1

        '''当前scale计算完，通过pooling缩减time维度为原来的一半'''
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    '''最后time缩减到1，再计算一次instance'''
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    
    '''loss在各个层次上进行平均'''
    return loss / d

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)  # 自己跟自己对比，loss=0

    '''合并为大矩阵计算相似度，对B处理'''
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    
    '''看AB和BA两个子矩阵，取这两个矩阵的对角线值，相加取log'''
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits+1e-10, dim=-1)
    print(torch.isnan(logits).int().sum())
    
    '''平均，取半'''
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    a = torch.isnan(loss)
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    
    '''合并为大矩阵计算相似度，对T处理'''
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T

    '''看AB和BA两个子矩阵，取这两个矩阵的对角线值，相加取log'''
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    '''平均，取半'''
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    a = torch.isnan(loss)
    return loss
