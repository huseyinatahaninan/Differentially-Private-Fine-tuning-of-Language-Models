import torch
import torch.nn as nn

import numpy as np

def process_batch_grad(batch_grad, scale):
    dim = len(batch_grad.shape)
    scale = scale.view([batch_grad.shape[0]] + [1]*(dim - 1))
    batch_grad.mul_(scale)
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

class LoraLinear(nn.Module):
    
    def __init__(self, indim, outdim, batch_dim=0):
        super(LoraLinear, self).__init__()

        self.batch_dim = batch_dim

        tensor = torch.ones(())
        self.weight = nn.Parameter(tensor.new_empty(size=(outdim, indim)))

        self.register_forward_hook(linear_forward_hook)
        self.register_backward_hook(linear_backward_hook)    

    def forward(self, x):
        acti = torch.matmul(x, self.weight.T)
        return acti
