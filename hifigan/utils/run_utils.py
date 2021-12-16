import argparse
import collections

import torch
from torch import nn

import hifigan.model as module_arch
from hifigan.trainer import Trainer
import hifigan.loss as module_loss
from hifigan.utils.parse_config import ConfigParser


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def prepare_trainer_for_training(config, dataloaders, device, device_ids,
                                 logger):
    generator = config.init_obj(config["archG"], module_arch)
    generator.apply(weights_init)
    logger.info(generator)
    # prepare for (multi-device) GPU training
    generator = generator.to(device)
    if len(device_ids) > 1:
        generator = torch.nn.DataParallel(generator, device_ids=device_ids)

    is_discriminator = (config.get("archD", None) is not None)
    if is_discriminator:
        discriminator = config.init_obj(config["archD"], module_arch)
        discriminator.apply(weights_init)
        logger.info(discriminator)
        # prepare for (multi-device) GPU training
        discriminator = discriminator.to(device)
        if len(device_ids) > 1:
            discriminator = torch.nn.DataParallel(discriminator,
                                                  device_ids=device_ids)
    else:
        discriminator = None

    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    metrics = []
    # build optimizer, learning rate scheduler. delete every lines
    # containing lr_scheduler for disabling scheduler
    trainable_paramsG = filter(lambda p: p.requires_grad, generator
                               .parameters())
    optimizerG = config.init_obj(config["optimizerG"], torch.optim,
                                 trainable_paramsG)
    if is_discriminator:
        trainable_paramsD = filter(lambda p: p.requires_grad, discriminator
                                   .parameters())
        optimizerD = config.init_obj(config["optimizerD"], torch.optim,
                                     trainable_paramsD)
    else:
        optimizerD = None
    if config["lr_schedulerG"]["use"] == "False":
        lr_schedulerG = None
    else:
        lr_schedulerG = config.init_obj(config["lr_schedulerG"],
                                        torch.optim.lr_scheduler, optimizerG)
    schedulerG_frequency_of_update = config["lr_schedulerG"]["frequency"]
    if is_discriminator:
        if config["lr_schedulerD"]["use"] == "False":
            lr_schedulerD = None
        else:
            lr_schedulerD = config.init_obj(config["lr_schedulerD"],
                                            torch.optim.lr_scheduler,
                                            optimizerD)
        schedulerD_frequency_of_update = config["lr_schedulerD"]["frequency"]
    else:
        lr_schedulerD = None
        schedulerD_frequency_of_update = None
    trainer = Trainer(
        generator,
        discriminator,
        loss_module,
        metrics,
        optimizerG,
        optimizerD,
        log_step=config["trainer"]["log_step"],
        fid_log_step=config["trainer"]["fid_log_step"],
        config=config,
        device=device,
        data_loader=dataloaders["train"],
        valid_data_loader=dataloaders["val"],
        lr_schedulerG=lr_schedulerG,
        lr_schedulerD=lr_schedulerD,
        schedulerG_frequency_of_update=schedulerG_frequency_of_update,
        schedulerD_frequency_of_update=schedulerD_frequency_of_update,
        len_epoch=config["trainer"].get("len_epoch", None)
    )
    return trainer


def run_main(main):
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given
    # in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float,
                   target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int,
            target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
