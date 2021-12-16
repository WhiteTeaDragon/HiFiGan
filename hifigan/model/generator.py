from torch import nn

from hifigan.base import BaseModel


class ResBlock(nn.Module):
    def __init__(self, d_r, k_r, channels):
        super(ResBlock, self).__init__()
        self.num_blocks = len(d_r)
        self.net = []
        for i in range(len(d_r)):
            for j in range(len(d_r[i])):
                self.net.append(nn.LeakyReLU(0.1))
                self.net.append(nn.Conv1d(channels, channels,
                                          kernel_size=(k_r, 1),
                                          dilation=d_r[i][j], padding="same"))
        self.len_block = len(self.net) // self.num_blocks
        self.net = nn.ModuleList(self.net)

    def forward(self, input_tensor):
        k = 0
        for i in range(self.num_blocks):
            x = input_tensor
            for j in range(self.len_block):
                x = self.net[k](x)
                k += 1
            input_tensor = input_tensor + x
        return input_tensor


class MRF(nn.Module):
    def __init__(self, k_r, d_r, channels):
        super(MRF, self).__init__()
        num_of_resblocks = len(k_r)
        self.blocks = []
        for i in range(num_of_resblocks):
            self.blocks.append(ResBlock(d_r[i], k_r[i], channels))
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, input_tensor):
        output = None
        for i in range(len(self.blocks)):
            x = self.blocks[i](input_tensor)
            if output is None:
                output = x
            else:
                output = output + x
        return output


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, k_r, d_r):
        super(Block, self).__init__()
        self.activation = nn.LeakyReLU(0.1)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, (kernel, 1),
                                       stride=stride)
        self.mrf = MRF(k_r, d_r, out_channels)

    def forward(self, input_tensor):
        return self.mrf(self.conv(self.activation(input_tensor)))


class Generator(BaseModel):
    def __init__(self, hidden_ch, k_u, k_r, d_r, d_r_repeat):
        super(Generator, self).__init__()
        d_r = [d_r] * d_r_repeat
        self.conv1 = nn.Conv1d(80, hidden_ch, (7, 1))
        self.layers = []
        in_channels = hidden_ch
        out_channels = in_channels
        for i in range(len(k_u)):
            out_channels = in_channels // 2
            self.layers.append(Block(in_channels, out_channels, k_u[i],
                                    k_u[i] // 2, k_r, d_r))
            in_channels = out_channels
        self.layers = nn.Sequential(*self.layers)
        self.activation1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(out_channels, 1, (7, 1))
        self.activation2 = nn.Tanh()

    def forward(self, melspec, *args, **kwargs):
        x = self.layers(self.conv1(melspec))
        x = self.activation2(self.conv2(self.activation1(x)))
        return {"output": x}
