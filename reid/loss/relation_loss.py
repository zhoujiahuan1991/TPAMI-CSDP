import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import math



def cosine_similarity(input1, input2):
    """Computes cosine distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = torch.mm(input1_normed, input2_normed.t())
    return distmat


def relation_kl_loss(x,y,tau=0.1):
    rel_matrix_1=cosine_similarity(x,x)
    rel_matrix_1=F.softmax(rel_matrix_1/tau, dim=1)

    rel_matrix_2=cosine_similarity(y,y)
    rel_matrix_2=F.softmax(rel_matrix_2/tau, dim=1)

    rel_matrix_1_log=torch.log(rel_matrix_1)
    kl_loss= F.kl_div(rel_matrix_1_log, rel_matrix_2)
    return kl_loss

def relation_js_loss(x,y,tau=0.1):
    js_loss=(relation_kl_loss(x,y,tau)+relation_kl_loss(y,x,tau))/2
    return js_loss