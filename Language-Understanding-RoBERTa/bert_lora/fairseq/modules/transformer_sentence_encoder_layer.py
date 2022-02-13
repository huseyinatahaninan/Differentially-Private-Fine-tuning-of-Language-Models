# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
)

from fairseq.lora_utils import LoraLinear

class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        args,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = 'relu',
        export: bool = False,
        encoder_normalize_before: bool = False,
    ) -> None:

        super().__init__()

        self.args = args

        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            args=self.args,
            dropout=attention_dropout,
            bias=True
        )

        
        self.normalize_before = encoder_normalize_before
        

        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # new added, lora layers 
        self.fc1_right = LoraLinear(self.embedding_dim, args.k)
        self.fc1_left = LoraLinear(args.k, ffn_embedding_dim)

        self.fc2_right = LoraLinear(ffn_embedding_dim, args.k)
        self.fc2_left = LoraLinear(args.k, self.embedding_dim)
        
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.reset_LoRA_parameters()

    def reset_LoRA_parameters(self):
        # reset LoRA parameters: in_proj_left and out_proj_left are initially set to 0.
        nn.init.constant_(self.fc1_left.weight, 0.)
        nn.init.constant_(self.fc2_left.weight, 0.)

        self.fc1_right.weight.data = self.fc1_right.weight.data.float()
        self.fc2_right.weight.data = self.fc2_right.weight.data.float()

        nn.init.xavier_normal_(self.fc1_right.weight)
        nn.init.xavier_normal_(self.fc2_right.weight)

        if(self.args.fp16):
            self.fc1_right.weight.data = self.fc1_right.weight.data.half()
            self.fc2_right.weight.data = self.fc2_right.weight.data.half()

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        rel_pos_bias: torch.Tensor = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x, _ = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
            rel_pos_bias=rel_pos_bias,
        )


        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)

        ############### LoRA forward step               
        lora_x = self.fc1_right(x)
        lora_x = self.fc1_left(lora_x)
        residual_x = self.fc1(x)
        x = lora_x + residual_x
        #################

        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)


        ################ LoRA forward step
        lora_x = self.fc2_right(x)
        lora_x = self.fc2_left(lora_x)
        residual_x = self.fc2(x)
        x = lora_x + residual_x
        ################

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        
        return x
    
    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x
