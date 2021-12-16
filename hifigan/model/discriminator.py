import torch
from torch import nn
from torch.nn.utils import weight_norm, spectral_norm

from hifigan.base import BaseModel


class DiscriminatorFromSubs(nn.Module):
    def __init__(self, discriminators):
        super(DiscriminatorFromSubs, self).__init__()
        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, target, model_output):
        target_res = []
        model_res = []
        target_features = []
        model_features = []
        for i in range(len(self.discriminators)):
            x, f = self.discriminators[i](target)
            target_res.append(x)
            target_features.append(f)
            x, f = self.discriminators[i](model_output)
            model_res.append(x)
            model_features.append(f)
        return target_res, model_res, torch.stack(target_features), \
            torch.stack(model_features)


class SubDiscriminator(nn.Module):
    def __init__(self, layers):
        super(SubDiscriminator, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        features = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            features.append(x)
        return x, torch.stack(features)


class PeriodSubDiscriminator(SubDiscriminator):
    def __init__(self, period):
        self.period = period
        layers = [nn.Sequential(
            weight_norm(nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(3, 1),
                                  padding=(2, 0))),
            nn.LeakyReLU(0.1)
        ), nn.Sequential(
            weight_norm(nn.Conv2d(32, 128, kernel_size=(5, 1), stride=(3, 1),
                                  padding=(2, 0))),
            nn.LeakyReLU(0.1)
        ), nn.Sequential(
            weight_norm(nn.Conv2d(128, 512, kernel_size=(5, 1), stride=(3, 1),
                                  padding=(2, 0))),
            nn.LeakyReLU(0.1)
        ), nn.Sequential(
            weight_norm(nn.Conv2d(512, 1024, kernel_size=(5, 1), stride=(3, 1),
                                  padding=(2, 0))),
            nn.LeakyReLU(0.1)
        ), nn.Sequential(
            weight_norm(nn.Conv2d(1024, 1024, kernel_size=(5, 1),
                                  padding=(2, 0))),
            nn.LeakyReLU(0.1)
        ), weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1),
                                 padding=(1, 0)))]
        super(PeriodSubDiscriminator, self).__init__(layers)

    def forward(self, input_tensor):
        batch_size, ch, t = input_tensor.shape
        assert ch == 1
        padding = (self.period - (t % self.period)) % self.period
        x = nn.functional.pad(input_tensor, (0, padding), "reflect")
        x = x.reshape(-1, 1, (t + self.period - 1) // self.period, self.period)
        return super(PeriodSubDiscriminator, self).forward(x)


class MPD(DiscriminatorFromSubs):
    def __init__(self, periods):
        discriminators = []
        for i in range(len(periods)):
            discriminators.append(PeriodSubDiscriminator(periods[i]))
        super(MPD, self).__init__(discriminators)


class ScaleSubDiscriminator(SubDiscriminator):
    def __init__(self, norm):
        layers = [
            nn.Sequential(
                norm(nn.Conv1d(1, 128, kernel_size=(15,), stride=(1,),
                               padding=7)),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                norm(nn.Conv1d(128, 128, kernel_size=(41,), stride=(2,),
                               groups=4, padding=20)),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                norm(nn.Conv1d(128, 256, kernel_size=(41,), stride=(2,),
                               groups=16, padding=20)),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                norm(nn.Conv1d(256, 512, kernel_size=(41,), stride=(4,),
                               groups=16, padding=20)),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                norm(nn.Conv1d(512, 1024, kernel_size=(41,), stride=(4,),
                               groups=16, padding=20)),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                norm(nn.Conv1d(1024, 1024, kernel_size=(41,), stride=(1,),
                               groups=16, padding=20)),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                norm(nn.Conv1d(1024, 1024, kernel_size=(5,), stride=(1,),
                               padding=2)),
                nn.LeakyReLU(0.1)
            ),
            norm(nn.Conv1d(1024, 1, kernel_size=(3,), stride=(1,),
                           padding=1))
        ]
        super(ScaleSubDiscriminator, self).__init__(layers)


class MSD(DiscriminatorFromSubs):
    def __init__(self):
        discriminators = [
            ScaleSubDiscriminator(spectral_norm),
            nn.Sequential(
                nn.AvgPool1d(4, 2, padding=2),
                ScaleSubDiscriminator(weight_norm)
            ),
            nn.Sequential(
                nn.AvgPool2d(4, 2, padding=2),
                ScaleSubDiscriminator(weight_norm)
            )
        ]
        super(MSD, self).__init__(discriminators)


class Discriminator(BaseModel):
    def __init__(self, periods):
        super(Discriminator, self).__init__()
        self.mpd = MPD(periods)
        self.msd = MSD()

    def forward(self, target, model_output):
        return {"mpd": self.mpd(target.unsqueeze(1), model_output),
                "msd": self.msd(target.unsqueeze(1), model_output)}
