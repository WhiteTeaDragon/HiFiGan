import librosa
import torch
import torchaudio
from torch.utils.data import DataLoader
from copy import copy

import hifigan
from hifigan.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    config_params = list(configs["data"].items())
    for i in range(len(config_params)):
        assert config_params[i][0] == "all" or config_params[i][0] == "test", \
            "Data type must be one all or one test"
        assert len(config_params) == 1, "With all specified -- use only" \
                                        " one dataset"
        params = config_params[i][1]
        num_workers = params.get("num_workers", 1)
        dataset = configs.init_obj(params["datasets"][0], hifigan.datasets,
                                   configs)
        if "test_size" in params:
            test_size = int(params["test_size"])
            train_size = len(dataset) - test_size
            train_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, test_size])
            test_dataset.dataset = copy(dataset)
            test_dataset.dataset.segment_size = None
            split = "train"
        else:
            train_dataset = dataset
            train_dataset.segment_size = None
            split = "test"
        # select batch size or batch sampler
        assert "batch_size" in params,\
            "You must provide batch_size for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = False
        else:
            raise Exception()
        train_dataloader = DataLoader(
            train_dataset, batch_size=bs, shuffle=shuffle,
            num_workers=num_workers)
        dataloaders[split] = train_dataloader
        if "test_size" in params:
            test_dataloader = DataLoader(
                test_dataset, batch_size=bs, shuffle=shuffle,
                num_workers=num_workers)
            dataloaders["val"] = test_dataloader
    return dataloaders


def initialize_mel_spec(config, device=None):
    sr = config["preprocessing"]["sr"]
    args = config["preprocessing"]["spectrogram"]["args"]
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=args["n_fft"],
        n_mels=args["n_mels"],
        fmin=args["f_min"],
        fmax=args["f_max"]
    ).T
    wave2spec = config.init_obj(
        config["preprocessing"]["spectrogram"],
        torchaudio.transforms,
        center=False
    )
    if device is not None:
        wave2spec = wave2spec.to(device)
    wave2spec.mel_scale.fb.copy_(torch.tensor(mel_basis))
    return wave2spec
