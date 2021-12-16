import torch
from torch import nn

from pix2pix.base import BaseModel
from pix2pix.model.generator import DownBlock


class Discriminator(BaseModel):
    def __init__(self, input_ch, output_ch, hidden_ch):
        super(Discriminator, self).__init__()
        self.down_blocks = nn.Sequential(
            DownBlock(input_ch + output_ch, hidden_ch, relu=False,
                      bnorm=False),
            DownBlock(hidden_ch, hidden_ch * 2),
            DownBlock(hidden_ch * 2, hidden_ch * 4),
            DownBlock(hidden_ch * 4, hidden_ch * 8, stride=1),
            DownBlock(hidden_ch * 8, 1, bnorm=False, stride=1),
            nn.Sigmoid()
        )

    def forward(self, input_image, output_image):
        input = torch.cat((input_image, output_image), dim=1)
        input = self.down_blocks(input)
        return input
