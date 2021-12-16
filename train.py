#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings

import numpy as np
import torch

from hifigan.datasets.utils import get_dataloaders
from hifigan.utils import prepare_device
from hifigan.utils.run_utils import prepare_trainer_for_training, run_main

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    device, device_ids = prepare_device(config["n_gpu"])
    # setup data_loader instances
    dataloaders = get_dataloaders(config, device)
    if config["overfit_on_one_batch"] == "True":
        dataloaders["train"] = [next(iter(dataloaders["train"]))]

    # build model architecture, then print to console
    trainer = prepare_trainer_for_training(config, dataloaders, device,
                                           device_ids, logger)

    trainer.train()


if __name__ == "__main__":
    run_main(main)
