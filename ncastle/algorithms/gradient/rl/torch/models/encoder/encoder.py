# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

class AttnHead(nn.Module):

    def __init__(self, hidden_dim, out_sz, device=None):

        super().__init__()
        self.device = device

        self._seq_fts = nn.Conv1d(in_channels=hidden_dim,
                                  out_channels=out_sz,
                                  kernel_size=(1,),
                                  bias=False,
                                  device=self.device)

        self._f_1 = nn.Conv1d(in_channels=out_sz,
                              out_channels=1,
                              kernel_size=(1,),
                                  device=self.device)
        self._f_2 = nn.Conv1d(in_channels=out_sz,
                              out_channels=1,
                              kernel_size=(1,),
                                  device=self.device)

        self._ret = nn.Conv1d(in_channels=hidden_dim,
                              out_channels=out_sz,
                              kernel_size=(1,),
                                  device=self.device)

        self._bias = nn.Parameter(torch.ones(1,device=self.device))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self._bias, 0)

    def forward(self, seq, in_drop=0.0, coef_drop=0.0, residual=False):

        seq = seq.permute(0,2,1)

        if in_drop != 0.0:
            seq = F.dropout(seq, in_drop)

        seq_fts = self._seq_fts(seq)

        # simplest self-attention possible
        f_1 = self._f_1(seq_fts)
        f_2 = self._f_2(seq_fts)

        logits = f_1 + f_2.permute(0, 2, 1)
        coefs = torch.softmax(F.leaky_relu(logits), dim=0)

        if coef_drop != 0.0:
            coefs = F.dropout(coefs, coef_drop)
        if in_drop != 0.0:
            seq_fts = F.dropout(seq_fts, in_drop)

        vals = torch.matmul(coefs.permute(0, 2, 1), seq_fts.permute(0, 2, 1))
        ret = vals + self._bias

        # residual connection
        if residual:
            if seq.permute(0, 2, 1).shape[-1] != ret.shape[-1]:
                ret = ret + self._ret(seq).permute(0, 2, 1)  # activation
            else:
                ret = ret + seq.permute(0, 2, 1)

        return F.elu(ret)  # activation


class GATEncoder(nn.Module):
 
    def __init__(self, batch_size , max_length, input_dimension, hidden_dim,
                 num_heads, num_stacks, residual, is_train, device=None):

        super().__init__()

        self.batch_size = batch_size # batch size
        self.max_length = max_length # input sequence length (number of cities)
        self.input_dimension = input_dimension # dimension of input, multiply 2 for expanding dimension to input complex value to tf, add 1 token
 
        self.hidden_dim = hidden_dim # dimension of embedding space (actor)
        self.num_heads = num_heads
        self.num_stacks = num_stacks
        self.residual = residual
        self.device = device
        self.is_training = is_train #not self.inference_mode

        self.head_hidden_dim = int(self.hidden_dim / self.num_heads)

        self.attn_head = AttnHead(
            self.hidden_dim, self.head_hidden_dim, device=self.device)

    def forward(self, inputs):
        """
        input shape: (batch_size, max_length, input_dimension)
        output shape: (batch_size, max_length, input_embed)
        """
        # First stack
        h_1 = inputs
        for _ in range(self.num_stacks):
            attns = []
            for _ in range(self.num_heads):
                attns.append(self.attn_head(h_1,  in_drop=0, coef_drop=0, residual=self.residual))
            h_1 = torch.cat(attns, axis=-1)

        return h_1


'''
Adapted from kyubyong park, June 2017.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
# Apply multihead attention to a 3d tensor with shape [batch_size, seq_length, n_hidden].
# Attention size = n_hidden should be a multiple of num_head
# Returns a 3d tensor with shape of [batch_size, seq_length, n_hidden]
class MultiheadAttention(nn.Module):
    
    def __init__(self, input_dimension, num_units=None, device=None):

        super().__init__()
        self.device = device
        # Linear projections
        # Q_layer = nn.Linear(in_features=input_dimension, out_features=num_units)
        self.Q_layer = nn.Sequential(nn.Linear(in_features=input_dimension, out_features=num_units),
                                nn.ReLU()).to(self.device)
        self.K_layer = nn.Sequential(nn.Linear(in_features=input_dimension, out_features=num_units),
                                nn.ReLU()).to(self.device)
        self.V_layer = nn.Sequential(nn.Linear(in_features=input_dimension, out_features=num_units),
                                    nn.ReLU()).to(self.device)
        
        # Normalize
        self.bn_layer = nn.BatchNorm1d(input_dimension).to(self.device)

    def forward(self, inputs, num_heads=16, dropout_rate=0.1, is_training=True):

        input_dimension = inputs.shape[1]
        inputs = inputs.permute(0,2,1)

        Q = self.Q_layer(inputs)  # [batch_size, seq_length, n_hidden]
        K = self.K_layer(inputs)  # [batch_size, seq_length, n_hidden]
        V = self.V_layer(inputs)  # [batch_size, seq_length, n_hidden]

        # Split and concat
        Q_ = torch.cat(torch.split(Q, int(input_dimension/num_heads), dim=2), dim=0)  # [batch_size, seq_length, n_hidden/num_heads]
        K_ = torch.cat(torch.split(K, int(input_dimension/num_heads), dim=2), dim=0)  # [batch_size, seq_length, n_hidden/num_heads]
        V_ = torch.cat(torch.split(V, int(input_dimension/num_heads), dim=2), dim=0)  # [batch_size, seq_length, n_hidden/num_heads]

        # Multiplication
        outputs = torch.matmul(Q_, K_.permute([0, 2, 1]))  # num_heads*[batch_size, seq_length, seq_length]
        
        # Scale
        outputs = outputs / (K_.shape[-1] ** 0.5)

        # Activation
        outputs = torch.softmax(outputs, dim=0)  # num_heads*[batch_size, seq_length, seq_length]

        # Dropouts
        outputs = F.dropout(outputs, p=dropout_rate, training=is_training)

        # Weighted sum
        outputs = torch.matmul(outputs, V_)  # num_heads*[batch_size, seq_length, n_hidden/num_heads]
        
        # Restore shape
        outputs = torch.cat(torch.split(outputs, int(outputs.shape[0]/num_heads), dim=0), dim=2)  # [batch_size, seq_length, n_hidden]
        
        # Residual connection
        outputs = outputs + inputs  # [batch_size, seq_length, n_hidden]

        outputs = outputs.permute(0,2,1)
        
        # Normalize
        outputs = self.bn_layer(outputs)  # [batch_size, seq_length, n_hidden]

        return outputs

class TransformerEncoder(nn.Module):
 
    def __init__(self, batch_size, max_length, input_dimension, hidden_dim,
                 num_heads, num_stacks, is_train, device):

        super().__init__()

        self.batch_size = batch_size  # batch size
        self.max_length = max_length  # input sequence length (number of cities)
        # dimension of input, multiply 2 for expanding dimension to input complex value to tf, add 1 token
        self.input_dimension = input_dimension

        self.input_embed = hidden_dim  # dimension of embedding space (actor)
        self.num_heads = num_heads
        self.num_stacks = num_stacks
        self.device = device
        self.is_training = is_train  # not self.inference_mode

        # self._emb_params = LayerParams(self, 'emb', self.device)
        self.emb = nn.Parameter(torch.Tensor(*(1, self.input_dimension,
                                               self.input_embed)).to(self.device))
        self.reset_parameters()

        # Batch Normalization
        self.bn_layer2 = nn.BatchNorm1d(self.input_dimension).to(self.device)

        # attention
        self.multihead_attention = []
        for i in range(self.num_stacks):  # num blocks
            multihead_attention = MultiheadAttention(self.input_dimension,
                                                     num_units=self.input_embed,
                                                     device=self.device)
            self.multihead_attention.append(multihead_attention)

        # FeedForward
        self.feedforward = []
        for i in range(self.num_stacks):  # num blocks
            feedforward = FeedForward(num_units=[4*self.input_embed, self.input_embed],
                                      device=self.device)
            self.feedforward.append(feedforward)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb)
 
    def forward(self, inputs):

        inputs = inputs.permute(0,2,1)

        # Embed input sequence
        # W_embed = self._emb_params.get_weights((1, self.input_dimension, self.input_embed))
        W_embed = self.emb

        # conv1 = nn.Conv1d(in_channels=self.input_dimension, out_channels=self.input_embed, kernel_size=1)
        # self.embedded_input = conv1(inputs)
        W_embed_ = W_embed.permute(2,1,0)
        # self.embedded_input = F.conv1d(inputs, W_embed_, stride=1, padding='valid')
        self.embedded_input = F.conv1d(inputs, W_embed_, stride=1)

        # Batch Normalization
        self.enc = self.bn_layer2(self.embedded_input)
        
        # Blocks
        for i in range(self.num_stacks):  # num blocks
            
            self.enc = self.multihead_attention[i](self.enc, num_heads=self.num_heads, 
                                                   dropout_rate=0.0, is_training=self.is_training)
            # Feed Forward
            self.enc = self.feedforward[i](self.enc)

        # Return the output activations [Batch size, Sequence Length, Num_neurons] as tensors.
        self.encoder_output = self.enc  ### NOTE: encoder_output is the ref for attention ###

        self.encoder_output = self.encoder_output.permute(0,2,1)

        return self.encoder_output

class FlashMultiheadAttention(nn.Module):
    def __init__(self, input_dimension, num_units=None, device=None):
        super().__init__()
        self.device = device
        
        # Linear transformations for Q, K, V
        self.Q_layer = nn.Sequential(nn.Linear(in_features=input_dimension, out_features=num_units),
                                nn.ReLU()).to(self.device)
        self.K_layer = nn.Sequential(nn.Linear(in_features=input_dimension, out_features=num_units),
                                nn.ReLU()).to(self.device)
        self.V_layer = nn.Sequential(nn.Linear(in_features=input_dimension, out_features=num_units),
                                    nn.ReLU()).to(self.device)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.1)
        
        # Output projection
        self.output_layer = nn.Linear(num_units, input_dimension).to(device)
        
        # Layer normalization
        self.norm = nn.LayerNorm(input_dimension).to(device)

    def flash_attention_forward(self, Q, K, V, mask=None):
        BLOCK_SIZE = 32
        NEG_INF = -torch.finfo(torch.float32).max  # -infinity
        EPSILON = 1e-10
        # mask = torch.randint(0, 2, (128, 8)).to(device='cuda')
        O = torch.zeros_like(Q, requires_grad=True)
        l = torch.zeros(Q.shape[:-1])[..., None]
        m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

        O = O.to(device='cuda')
        l = l.to(device='cuda')
        m = m.to(device='cuda')

        Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
        KV_BLOCK_SIZE = BLOCK_SIZE

        Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
        K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
        V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
        if mask is not None:
            mask_BLOCKS = list(torch.split(mask, KV_BLOCK_SIZE, dim=1))

        Tr = len(Q_BLOCKS)
        Tc = len(K_BLOCKS)

        O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
        l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
        m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

        for j in range(Tc):
            Kj = K_BLOCKS[j]
            Vj = V_BLOCKS[j]
            if mask is not None:
                maskj = mask_BLOCKS[j]

            for i in range(Tr):
                Qi = Q_BLOCKS[i]
                Oi = O_BLOCKS[i]
                li = l_BLOCKS[i]
                mi = m_BLOCKS[i]

                scale = 1 / np.sqrt(Q.shape[-1])
                Qi_scaled = Qi * scale

                S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi_scaled, Kj)
                if mask is not None:
                    # Masking
                    maskj_temp = rearrange(maskj, 'b j -> b 1 1 j')
                    S_ij = torch.where(maskj_temp > 0, S_ij, NEG_INF)

                m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
                P_ij = torch.exp(S_ij - m_block_ij)
                if mask is not None:
                    # Masking
                    P_ij = torch.where(maskj_temp > 0, P_ij, 0.)

                l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON

                P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

                mi_new = torch.maximum(m_block_ij, mi)
                li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij

                O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi + (
                        torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
                l_BLOCKS[i] = li_new
                m_BLOCKS[i] = mi_new

        O = torch.cat(O_BLOCKS, dim=2)
        l = torch.cat(l_BLOCKS, dim=2)
        m = torch.cat(m_BLOCKS, dim=2)
        return O

    def forward(self, inputs, num_heads=16, dropout_rate=0.1, is_training=True):
        inputs = inputs.permute(0,2,1)
        batch_size, seq_length, _ = inputs.shape
        
        # Compute Q, K, V
        Q = self.Q_layer(inputs).view(batch_size, seq_length, num_heads, -1).permute(0, 2, 1, 3)
        K = self.K_layer(inputs).view(batch_size, seq_length, num_heads, -1).permute(0, 2, 1, 3)
        V = self.V_layer(inputs).view(batch_size, seq_length, num_heads, -1).permute(0, 2, 1, 3)
        
        # Apply FlashAttention
        outputs = self.flash_attention_forward(Q, K, V)
        
        # Restore shape
        outputs = outputs.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
        
        # Output projection
        outputs = self.output_layer(outputs)
        
        # Residual connection and normalization
        outputs = self.norm(outputs + inputs)
        
        outputs = outputs.permute(0,2,1)
        
        return outputs

class FlashformerEncoder(nn.Module):
 
    def __init__(self, batch_size, max_length, input_dimension, hidden_dim,
                 num_heads, num_stacks, is_train, device):

        super().__init__()

        self.batch_size = batch_size  # batch size
        self.max_length = max_length  # input sequence length (number of cities)
        # dimension of input, multiply 2 for expanding dimension to input complex value to tf, add 1 token
        self.input_dimension = input_dimension

        self.input_embed = hidden_dim  # dimension of embedding space (actor)
        self.num_heads = num_heads
        self.num_stacks = num_stacks
        self.device = device
        self.is_training = is_train  # not self.inference_mode

        # self._emb_params = LayerParams(self, 'emb', self.device)
        self.emb = nn.Parameter(torch.Tensor(*(1, self.input_dimension,
                                               self.input_embed)).to(self.device))
        self.reset_parameters()

        # Batch Normalization
        self.bn_layer2 = nn.BatchNorm1d(self.input_dimension).to(self.device)

        # attention
        self.multihead_attention = []
        for i in range(self.num_stacks):  # num blocks
            multihead_attention = FlashMultiheadAttention(self.input_dimension,
                                                     num_units=self.input_embed,
                                                     device=self.device)
            self.multihead_attention.append(multihead_attention)

        # FeedForward
        self.feedforward = []
        for i in range(self.num_stacks):  # num blocks
            feedforward = FeedForward(num_units=[4*self.input_embed, self.input_embed],
                                      device=self.device)
            self.feedforward.append(feedforward)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb)
 
    def forward(self, inputs):

        inputs = inputs.permute(0,2,1)

        # Embed input sequence
        # W_embed = self._emb_params.get_weights((1, self.input_dimension, self.input_embed))
        W_embed = self.emb

        # conv1 = nn.Conv1d(in_channels=self.input_dimension, out_channels=self.input_embed, kernel_size=1)
        # self.embedded_input = conv1(inputs)
        W_embed_ = W_embed.permute(2,1,0)
        # self.embedded_input = F.conv1d(inputs, W_embed_, stride=1, padding='valid')
        self.embedded_input = F.conv1d(inputs, W_embed_, stride=1)

        # Batch Normalization
        self.enc = self.bn_layer2(self.embedded_input)
        
        # Blocks
        for i in range(self.num_stacks):  # num blocks
            
            self.enc = self.multihead_attention[i](self.enc, num_heads=self.num_heads, 
                                                   dropout_rate=0.0, is_training=self.is_training)
            # Feed Forward
            self.enc = self.feedforward[i](self.enc)

        # Return the output activations [Batch size, Sequence Length, Num_neurons] as tensors.
        self.encoder_output = self.enc  ### NOTE: encoder_output is the ref for attention ###

        self.encoder_output = self.encoder_output.permute(0,2,1)

        return self.encoder_output

# Apply point-wise feed forward net to a 3d tensor with shape [batch_size, seq_length, n_hidden]
# Returns: a 3d tensor with the same shape and dtype as inputs
class FeedForward(nn.Module):

    def __init__(self, num_units=[2048, 512], device=None):

        super().__init__()
        self.device = device
        # Inner layer
        self.conv1 = nn.Conv1d(in_channels=num_units[1],
                               out_channels=num_units[0],
                               kernel_size=(1,),
                               bias=True).to(self.device)
        # Readout layer
        self.conv2 = nn.Conv1d(in_channels=num_units[0],
                               out_channels=num_units[1],
                               kernel_size=(1,),
                               bias=True).to(self.device)

        self.bn_layer1 = nn.BatchNorm1d(num_units[1]).to(self.device)  # 传入通道数

    def forward(self, inputs):

        outputs = self.conv1(inputs)
        outputs = F.relu(outputs)

        outputs = self.conv2(outputs)

        # Residual connection
        outputs += inputs

        outputs = self.bn_layer1(outputs)  # [batch_size, seq_length, n_hidden]

        return outputs



