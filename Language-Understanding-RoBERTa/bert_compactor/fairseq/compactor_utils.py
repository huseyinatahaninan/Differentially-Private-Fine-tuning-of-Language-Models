import torch
import torch.nn as nn

import numpy as np
import math

def process_batch_grad(batch_grad, scale):
    dim = len(batch_grad.shape)
    scale = scale.view([batch_grad.shape[0]] + [1]*(dim - 1))
    batch_grad.mul_(scale) # gradient clipping
    batch_g = torch.sum(batch_grad, dim=0)
    return batch_g 



def linear_forward_hook(module, intsr, outtsr):
    module.input = intsr[0].detach()

def linear_backward_hook(module, grad_input, grad_output):

    grad_output = grad_output[0].detach() # len, n, outdim
    grad_input = module.input #len, n, indim


    if(len(grad_output.shape)==3): # normal layers
        grad_output = grad_output.permute(1, 2, 0) # n, outdim, len
        grad_input = grad_input.permute(1, 0, 2) # n, len, indim

        module.weight.batch_grad = torch.bmm(grad_output, grad_input)
        

        if(hasattr(module, 'bias')):
            module.bias.batch_grad = torch.sum(grad_output, dim=2)

    elif(len(grad_output.shape)==2): #final classification layer
        grad_output = grad_output.view(grad_output.shape[0], grad_output.shape[1], 1) # n, outdim, 1
        grad_input = grad_input.view(grad_input.shape[0], 1, grad_input.shape[1]) # n, 1, indim

        module.weight.batch_grad = torch.bmm(grad_output, grad_input) 

        if(hasattr(module, 'bias')):
            module.bias.batch_grad = grad_output.view(grad_output.shape[0], grad_output.shape[1]) 

    else:
        raise 'not implemented error'

def phm_linear_forward_hook(module, intsr, outtsr):
    module.input = intsr[0].detach()


def phm_linear_backward_hook(module, grad_input, grad_output):
    
    grad_output = grad_output[0].detach() # len, btahcsize, ensemble, dim0, dim1
    grad_input = module.input # len, btahcsize, ensemble, dim2, dim3

    grad_input = grad_input.permute(0, 1, 2, 4, 3)
    unprocessed_grad = torch.matmul(grad_input, grad_output)
    module.weight.batch_grad = torch.sum(unprocessed_grad, dim=0) # sum over sentence length
    
    #print('weight size:', module.weight.shape, 'batch grad shape:', module.weight.batch_grad.shape, 'input shape:', grad_input.shape, 'output shape:', grad_output.shape)

class PHMLinear_inner(nn.Module):
    def __init__(self, ensemble, indim, outdim):
        super(PHMLinear_inner, self).__init__()
        tensor = torch.ones(())
        self.weight = nn.Parameter(tensor.new_empty(size=(ensemble, indim, outdim)))
        torch.nn.init.xavier_normal_(self.weight, gain=math.sqrt(2))

    def forward(self, x):
        
        acti = torch.matmul(x, self.weight)
        return acti

def register_batch_mm_hook(module):
    module.register_forward_hook(phm_linear_forward_hook)
    module.register_backward_hook(phm_linear_backward_hook)  

class PHMLinear(nn.Module):
    
    def __init__(self, indim, outdim, n, rank=1): 
        # n: number of Kronecker products
        super(PHMLinear, self).__init__()

        assert indim%n == 0
        assert outdim%n == 0

        left_in_dim = n
        left_out_dim = n

        right_in_dim = indim // n
        right_out_dim = outdim // n

        num_left_param = n*n
        num_right_param = right_in_dim*right_out_dim

        self.left_inner = PHMLinear_inner(n, left_in_dim, left_out_dim)

        # we reparametrize the right layer
        self.right_inner_left = PHMLinear_inner(n, right_in_dim, rank)
        self.right_inner_right = PHMLinear_inner(n, rank, right_out_dim)

        register_batch_mm_hook(self.left_inner)
        register_batch_mm_hook(self.right_inner_left)
        register_batch_mm_hook(self.right_inner_right)

        self.left_in_dim = left_in_dim
        self.left_out_dim = left_out_dim
        self.right_in_dim = right_in_dim
        self.right_out_dim = right_out_dim
        self.n = n

    def forward(self, x):
        assert len(x.shape) == 3 # [sentence len., # of sentences, embed dim]
        orig_shape = x.shape

        # the following computation equivalents to a normal PHM layer
        # with this implementation, we can place the all the weights in nn.Linear modules and use customed hookers to compute per-example gradients
        x = x.view(*x.shape[0:2], 1, self.left_in_dim, self.right_in_dim)
        inner_acti = self.right_inner_left(x)
        inner_acti = self.right_inner_right(inner_acti)
        inner_acti = torch.transpose(inner_acti, -2, -1)
        outer_acti = self.left_inner(inner_acti)
        acti = outer_acti.view(*orig_shape[0:2], self.n, -1)

        acti = torch.sum(acti, dim=-2)
        return acti
