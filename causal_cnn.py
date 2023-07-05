"""
Implementation of causal CNNs partly taken and modified from
https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
"""

import torch


class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.

    @param chomp_size Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):

        return x[:, :, :-self.chomp_size]

class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)

class CasualConvolutionBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,dilation,final=False):
        super().__init__()
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(  # input(batch_size,feature_dim,time_dim)
            in_channels, out_channels, kernel_size,
            dilation=dilation
        ))
        relu1 = torch.nn.LeakyReLU(inplace=True)
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(  # input(batch_size,feature_dim,time_dim)
            in_channels, out_channels, kernel_size,
            dilation=dilation
        ))
        relu2 = torch.nn.LeakyReLU(inplace=True)

        self.causal = torch.nn.Sequential(
            conv1,  relu1, conv2, relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU(inplace=True) if final else None

    def forward(self,x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param padding Zero-padding applied to the left of the input of the
           non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = ((kernel_size - 1) * dilation) #// 2
        #print("padding",padding)
        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(            #input(batch_size,feature_dim,time_dim)
            in_channels, out_channels, kernel_size,
            dilation=dilation
            ,padding=padding
        ))
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU(inplace=True)

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation
            ,padding=padding
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU(inplace=True)

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1,
            chomp1,
            relu1, conv2,
            chomp2,
            relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU(inplace=True) if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)

class CausalConvTransblock(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 Lin,
                 time_dim,
                 kernel_size,
                 dilation):
        super(CausalConvTransblock, self).__init__()



        stride=3
        padding=((Lin-1)*stride+dilation*(kernel_size-1)+1-time_dim)/2
        padding=int(padding)
            #padding = dilation * (kernel_size - 1)
            #stride = (time_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / (Lin - 1)
            #stride=int(stride)


        #first deconv
        self.deconv = torch.nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, stride=stride
        )

        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.deconv(x)

        out = self.relu(out)
        #print("deconv:",out.shape)
        return out










class CausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size,time_dim,hidden_time_dim):
        super(CausalCNN, self).__init__()

        linear_proj = torch.nn.Linear(time_dim,hidden_time_dim)
        layers = []  # List of causal convolution blocks
        # layers+=[linear_proj]
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        #x = torch.tensor(x, dtype=torch.float32)
        return self.network(x)


class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).

    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size):
        super(CausalCNNEncoder, self).__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear = torch.nn.Linear(reduced_size, out_channels)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze, linear
        )

    def forward(self, x):
        return self.network(x)

def pass_through(x):
    return x

class inceptionblock(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernels = [2,4,8],bottleneck_channels=32):
        super(inceptionblock, self).__init__()
        self.conv_list=torch.nn.ModuleList()

        # input(batch_size,feature_dim,time_dim)->(batch_size,hidden_dim,time_dim)
        if in_channels > 1:
            self.bottleneck = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1
        for kernel_size in kernels:
            self.conv_list.append(torch.nn.utils.weight_norm(torch.nn.Conv1d(
                                in_channels=bottleneck_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                padding = (kernel_size-1)//2,
                                bias=False
                                )))
        self.relu = torch.nn.LeakyReLU()
        self.batchnorm=torch.nn.modules.BatchNorm1d(out_channels * (len(kernels) + 1), eps=1e-5)


        self.pooling=torch.nn.MaxPool1d(
            kernel_size=3,
            stride=1,
            padding=1)
        self.lastconv=torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
                                      )


    def forward(self,x):

        middle_output=[]
        bx=self.bottleneck(x)
        for conv in self.conv_list:
            middle_output.append(conv(bx))
        middle_output.append(self.lastconv(self.pooling(x)))
        #print(len(middle_output),middle_output[0].shape)

        output=self.relu(torch.cat(middle_output,axis=1))
        #print(output.shape)


        output=self.relu(self.batchnorm(output))
        return output

class Inception(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernels = [2,4,8],bottleneck_channels=32,depth=3):
        super(Inception, self).__init__()
        layers = []
        for i in range(depth):
            layers+=[inceptionblock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernels=kernels,
                bottleneck_channels=bottleneck_channels
            )]

        self.inception=torch.nn.Sequential(*layers)



        self.relu=torch.nn.LeakyReLU()

        # Residual connection
        realout_channels=out_channels * (len(kernels) + 1)

        self.upordownsample = torch.nn.Conv1d(
            in_channels, realout_channels, 1
        )

    def forward(self,x):
        middle=self.inception(x)
        res = self.upordownsample(x)
        output=self.relu( middle + res)
        return output








