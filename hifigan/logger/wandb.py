from datetime import datetime

import numpy as np
import wandb


class WanDBWriter:
    def __init__(self, config):
        self.mode = None
        self.step = None
        self.timer = None
        self.writer = None
        self.selected_module = ""

        wandb.login()

        if config['trainer'].get('wandb_project') is None:
            raise ValueError("please specify project name for wandb")

        wandb.init(
            project=config['trainer'].get('wandb_project'),
            config=config.config
        )
        self.wandb = wandb

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    def scalar_name(self, scalar_name):
        return f"{scalar_name}_{self.mode}"

    def add_scalar(self, scalar_name, scalar):
        self.wandb.log({
            self.scalar_name(scalar_name): scalar,
        }, step=self.step)

    def add_image(self, scalar_name, image, normalize=False):
        vis_util = wandb.util.get_module(
            "torchvision.utils", "torchvision is required to render images"
        )
        data = vis_util.make_grid(image, normalize=normalize)
        pil_image = wandb.util.get_module(
            "PIL.Image",
            required='wandb.Image needs the PIL package. To get it, run '
                     '"pip install pillow".',
        )
        _image = pil_image.fromarray(
            data.mul(255).clamp(0, 255).byte().permute(1, 2,
                                                       0).cpu().numpy()
        )
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Image(_image)
        }, step=self.step)

    def add_histogram(self, scalar_name, hist, bins=None):
        hist = hist.detach().cpu().numpy()
        np_hist = np.histogram(hist, bins=bins)
        if np_hist[0].shape[0] > 512:
            np_hist = np.histogram(hist, bins=512)

        hist = self.wandb.Histogram(
            np_histogram=np_hist
        )

        self.wandb.log({
            self.scalar_name(scalar_name): hist
        }, step=self.step)

    def add_audio(self, scalar_name, audio, sample_rate=None):
        audio = audio.detach().cpu().numpy().T
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Audio(
                audio, sample_rate=sample_rate)
        }, step=self.step)
