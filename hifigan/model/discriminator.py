from hifigan.base import BaseModel


class Discriminator(BaseModel):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, input_image, output_image):
        raise NotImplementedError
