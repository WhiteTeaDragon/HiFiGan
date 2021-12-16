from torch import nn
from torch.nn.utils import weight_norm

from hifigan.base import BaseModel


class SubDiscriminator(nn.Module):
    def __init__(self, period):
        super(SubDiscriminator, self).__init__()
        self.convs = []
        input_ch = 1
        output_ch = 32
        for i in range(4):
            output_ch *= 2
            self.convs.append(weight_norm(nn.Conv2d(input_ch, output_ch,
                                                    kernel_size=(5, 1),
                                                    stride=(3, 1),
                                                    padding=(2, 0))))
            input_ch = output_ch
            self.convs.append(nn.LeakyReLU(0.1))
        self.convs.append(weight_norm(nn.Conv2d(output_ch, 1024,
                                                kernel_size=(5, 1),
                                                padding=(2, 0))))
        self.convs.append(weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1),
                                                padding=(1, 0))))
        self.convs = nn.Sequential(*self.convs)

    def forward(self, input_tensor):
        batch_size, ch, t = input_tensor.shape
        assert ch == 1
        padding = (self.period - (t % self.period)) % self.period
        x = nn.functional.pad(input_tensor, (0, padding), "reflect")
        x = x.reshape(-1, 1, (t + self.period - 1) // self.period, self.period)
        x = self.convs(x)
        return x


class MPD(nn.Module):
    def __init__(self, periods):
        super(MPD, self).__init__()
        self.discriminators = []
        for i in range(len(periods)):
            self.discriminators.append(SubDiscriminator(periods[i]))
        self.discriminators = nn.ModuleList(self.discriminators)

    def forward(self, input_tensor):

        return input_tensor


class Discriminator(BaseModel):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, input_image, output_image):
        raise NotImplementedError
