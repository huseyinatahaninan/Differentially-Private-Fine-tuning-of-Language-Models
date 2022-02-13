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

from fairseq.adapter_utils import AdapterLinear

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
        # new added
        encoder_normalize_before: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters

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

        # new added, low-rank reparameterization for the FFN layer 
        self.normalize_before = encoder_normalize_before
        

        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)


        self.fc_right = AdapterLinear(self.embedding_dim, args.k)
        self.fc_left = AdapterLinear(args.k, self.embedding_dim)

        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.reset_Adapter_parameters()

    def reset_Adapter_parameters(self):
        # in_proj_left and out_proj_left are initially set to 0.
        nn.init.constant_(self.fc_left.weight, 0.)

        self.fc_right.weight.data = self.fc_right.weight.data.float()

        nn.init.xavier_normal_(self.fc_right.weight)

        if(self.args.fp16):
            self.fc_right.weight.data = self.fc_right.weight.data.half()

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
        # new added
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

        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        ################ Adapter forward step
        adapter_x = x
        adapter_x = self.fc_right(adapter_x)
        adapter_x = F.gelu(adapter_x)
        adapter_x = self.fc_left(adapter_x)
        x = x + adapter_x
        ################

        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        
        return x
    
    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x
