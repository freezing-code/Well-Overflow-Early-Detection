import torch
from torch import nn, Tensor
import math
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d
from typing import Optional
import causal_cnn


class linear_add_layer(nn.Module):

    def __init__(self, batch_size, hidden_dim):
        """

        :param batch_size:
        :param hidden_dim:
        """
        super(linear_add_layer, self).__init__()
        self.g = torch.nn.GELU()
        # self.bn=nn.BatchNorm1d(num_features=hidden_dim)
        # self.params = nn.ParameterList([nn.Parameter(torch.randn(batch_size, hidden_dim))for i in  range(2)])

    def forward(self, global_f, local_f):
        x = global_f + local_f
        # x = self.g(x)
        return x


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, net):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.net = net

    def forward(self, x, **kwargs):
        #    x = torch.tensor(x, dtype=torch.float32)
        return self.net(self.norm(x), **kwargs)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.num_heads = num_heads
        self.scale = dim_per_head ** -0.5

        inner_dim = dim_per_head * num_heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.attend = nn.Softmax(dim=-1)

        project_out = not (num_heads == 1 and dim_per_head == dim)
        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, l, d = x.shape

        '''i. QKV projection'''
        # (b,l,dim_all_heads x 3)

        qkv = self.to_qkv(x)

        # (3,b,num_heads,l,dim_per_head)
        qkv = qkv.view(b, l, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # 3 x (1,b,num_heads,l,dim_per_head)
        q, k, v = qkv.chunk(3)
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        '''ii. Attention computation'''
        attn = self.attend(
            torch.matmul(q, k.transpose(-1, -2)) * self.scale
        )

        '''iii. Put attention on Value & reshape'''
        # (b,num_heads,l,dim_per_head)
        z = torch.matmul(attn, v)
        # (b,num_heads,l,dim_per_head)->(b,l,num_heads,dim_per_head)->(b,l,dim_all_heads)
        z = z.transpose(1, 2).reshape(b, l, -1)
        # assert z.size(-1) == q.size(-1) * self.num_heads

        '''iv. Project out'''
        # (b,l,dim_all_heads)->(b,l,dim)
        out = self.out(z)
        # assert out.size(-1) == d

        return out


class Transformer(nn.Module):
    def __init__(self, dim, mlp_dim, feature_dim, depth=6, num_heads=8, dropout=0.3, cuda=False):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, num_heads=num_heads, dropout=dropout)),
                # SelfAttention(dim, num_heads=num_heads, dropout=dropout),
                PreNorm(dim, FFN(dim, mlp_dim, dropout=dropout))
                # FFN(dim, mlp_dim, dropout=dropout),
                # nn.BatchNorm1d(feature_dim),
            ]))

    def forward(self, x):
        # print(x.shape)
        for norm_attn, norm_ffn in self.layers:
            x = x + norm_attn(x)

            x = x + norm_ffn(x)
            # x = x + b2(x)

        return x


class positional_encoding(nn.Module):
    """
    shape=(B,C,L）
    """

    def __init__(self, shape, dropout=0.1, cuda=False):
        super().__init__()
        C = shape[1]
        L = shape[2]
        # self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(C, L)

        position = torch.arange(0, C, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, L, 2).float() * (-math.log(10000.0) / L))
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1,max_len, d_model]
        if cuda:
            pe = pe.cuda()
        print(pe.device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.size())
        # print(self.pe.size())
        pe = self.pe

        pe = pe.to(x.device)
        x = x + self.pe  # [:x.size(), :]
        return x


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class MHA(nn.modules.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MHA, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = Dropout(dropout)
        self.norm1 = BatchNorm1d(d_model, eps=1e-5)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)

        return src


class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                         dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim].
        # padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention,
        # TransformerEncoderLayer
        output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)

        return output


class TSTransformerdecoder(nn.Module):
    def __init__(self, feat_dim, d_model, dropout=0.1):
        super(TSTransformerdecoder, self).__init__()
        self.output_layer = nn.Linear(d_model, feat_dim)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, X):
        output = self.dropout1(X)
        output = self.output_layer(output)
        return output


class TST(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TST, self).__init__()

        self.TSTencoder = TSTransformerEncoder(feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward,
                                               dropout,
                                               pos_encoding, activation, norm, freeze)
        self.TSTdecoder = TSTransformerdecoder(feat_dim, d_model)

    def forward(self, X):
        output = self.TSTencoder(X)
        output = self.TSTdecoder(output)
        return output

class linearMT(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, kernel_size=3, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', cnndepth=3, cnnhidden_channels=64, freeze=False):
        super(linearMT, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)

        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                         dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        # inp = X.permute(1, 0, 2)
        inp = self.project_inp(X) * math.sqrt(
            self.d_model)  # [batch_size,seq_length,d_model]

        inp = inp.permute(1, 0,
                          2)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space

        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)

        return output

class MT(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, kernel_size=3, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', cnndepth=3, cnnhidden_channels=64, freeze=False):
        super(MT, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        #self.project_inp = nn.Linear(feat_dim, d_model)
        self.project_inp = causal_cnn.CausalCNN(in_channels=feat_dim, channels=cnnhidden_channels, out_channels=d_model,
                                                depth=cnndepth, kernel_size=kernel_size)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                         dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        # inp = X.permute(1, 0, 2)
        inp = X.permute(0, 2, 1)  # [batch_size,feat_dim,seq_length]
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [batch_size,d_model,ht]

        inp = inp.permute(2, 0,
                          1)  # [ht, batch_size, d_model] project input vectors to d_model dimensional space

        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp)  # (ht, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, ht, d_model)

        return output


class MIT(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, kernels=[2, 4, 8], dropout=0.1,
                 pos_encoding='fixed', activation='gelu', cnndepth=3, cnnhidden_channels=64, freeze=False):
        super(MIT, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        # self.project_inp = nn.Linear(feat_dim, d_model)
        # (batch_size,featuredim,timedim)->(batchsize,d_model x (kernelsize +1),timedim)
        self.project_inp = causal_cnn.Inception(in_channels=feat_dim, out_channels=d_model, kernels=kernels,
                                                bottleneck_channels=cnnhidden_channels, depth=cnndepth)
        # CausalCNN(in_channels=feat_dim,channels=cnnhidden_channels,out_channels=d_model,depth=cnndepth,kernel_size=kernel_size)
        self.d_model = d_model * (len(kernels) + 1)
        d_model = self.d_model
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                         dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        # inp = X.permute(1, 0, 2)
        inp = X.permute(0, 2, 1)  # [batch_size,feat_dim,seq_length]

        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [batch_size,d_model,seq_length]

        inp = inp.permute(2, 0,
                          1)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space

        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)

        return output


class MCA(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, kernel_size=3, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', cnndepth=3, cnnhidden_channels=64, freeze=False):
        super(MCA, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        # self.project_inp = nn.Linear(feat_dim, d_model)
        self.project_inp = causal_cnn.CausalCNN(in_channels=feat_dim, channels=cnnhidden_channels, out_channels=d_model,
                                                depth=cnndepth, kernel_size=kernel_size)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)
        mha = MHA(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.encoder = torch.nn.TransformerEncoder(mha, num_layers)
        self.act = _get_activation_fn(activation)

        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)
        self.norm = BatchNorm1d(d_model, eps=1e-5)
        self.dropout = Dropout(dropout)

    def forward(self, X):  # (batch_size,seq_length,feat_dim]
        inp = X.permute(0, 2, 1)  # [batch_size,feat_dim,seq_length]

        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [batch_size,d_model,seq_length]

        inp = inp.permute(2, 0,
                          1)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space

        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        # restore (seq_len, batch_size, d_model)
        # output2=self.linear2(self.dropout(self.act(self.linear1(output))))
        # output2=self.dropout(output)+output2
        # src = output2.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = self.norm(src)
        # src = src.permute(0,2,1) #(batch_size,seq_len,d_model

        src = output.permute(1, 0, 2)
        return src


class MTSTco(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, cnndepth, cnnhidden_channels,
                 kernel_size, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', freeze=False):
        super(MTSTco, self).__init__()
        # original is MT
        self.TSTencoder = MT(feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, kernel_size, dropout,
                             pos_encoding, activation, cnndepth, cnnhidden_channels, freeze)
        self.TSTdecoder = TSTransformerdecoder(feat_dim, d_model)

    def forward(self, X):
        """

        :param X:
        :return: output1:embedding,output2:classification result
        """
        output1 = self.TSTencoder(X)
        output2 = self.TSTdecoder(output1)
        return output1, output2


class MTST(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, cnndepth, cnnhidden_channels,
                 kernel_size, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', freeze=False):
        super(MTST, self).__init__()

        self.TSTencoder = MT(feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, kernel_size, dropout,
                             pos_encoding, activation, cnndepth, cnnhidden_channels, freeze)
        self.TSTdecoder = TSTransformerdecoder(feat_dim, d_model)

    def forward(self, X):
        output = self.TSTencoder(X)
        output = self.TSTdecoder(output)
        return output


class MTSCA(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, kernel_size=3, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', cnndepth=3, cnnhidden_channels=64, freeze=False):
        super(MTSCA, self).__init__()
        self.TSTencoder = MCA(feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, kernel_size, dropout,
                              pos_encoding, activation, cnndepth, cnnhidden_channels, freeze)
        self.TSTdecoder = TSTransformerdecoder(feat_dim, d_model)

    def forward(self, x):
        output = self.TSTencoder(x)

        output = self.TSTdecoder(output)
        return output


class MITST(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, cnndepth, cnnhidden_channels,
                 kernels, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', freeze=False):
        super(MITST, self).__init__()

        self.TSTencoder = MIT(feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, kernels, dropout,
                              pos_encoding, activation, cnndepth, cnnhidden_channels, freeze)
        # (batch_size,featuredim,timedim)->
        d_model = d_model * (len(kernels) + 1)
        self.TSTdecoder = TSTransformerdecoder(feat_dim, d_model)

    def forward(self, X):
        output = self.TSTencoder(X)
        output = self.TSTdecoder(output)
        return output
