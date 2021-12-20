import argparse
import json
import os
from pathlib import Path

import gdown as gdown
import torch
import torchaudio
from tqdm import tqdm

import hifigan.model as module_model
from hifigan.datasets.utils import get_dataloaders
from hifigan.trainer import Trainer
from hifigan.utils import ROOT_PATH
from hifigan.utils.parse_config import ConfigParser

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture
    model = config.init_obj(config["archG"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    audios = []
    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)
            batch["melspec"], _ = Trainer.get_spectrogram(batch["audio"])
            batch["device"] = device
            output = model(**batch)
            audios.append(output)
    out_file = Path(out_file)
    out_file.mkdir(exist_ok=True, parents=True)
    for i in range(len(audios)):
        path = out_file / f"{i}.wav"
        sr = config["preprocessing"]["sr"]
        torchaudio.save(path, audios[i].cpu(), sr)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="test_output",
        type=str,
        help="Folder to write results",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if args.resume is None:
        url = "https://drive.google.com/uc?id" \
              "=1r79iZvKknAtpnRuv1F_QjhcTlXfRAFU9"
        path_to_checkpoint = str(DEFAULT_CHECKPOINT_PATH.absolute().resolve())
        checkpoint_dir = ROOT_PATH / "default_test_model"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        if not Path(path_to_checkpoint).exists():
            gdown.download(url, path_to_checkpoint, quiet=False)
    else:
        path_to_checkpoint = args.resume

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(path_to_checkpoint).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=path_to_checkpoint)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["num_workers"] = args.jobs

    main(config, args.output)
