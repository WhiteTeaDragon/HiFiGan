import torch
from torch.utils.data import DataLoader

import hifigan
from hifigan.collate_fn.collate import collate_fn
from hifigan.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser, device):
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
            split = "train"
        else:
            train_dataset = dataset
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
            train_dataset, batch_size=bs, collate_fn=collate_fn,
            shuffle=shuffle, num_workers=num_workers)
        dataloaders[split] = train_dataloader
        if "test_size" in params:
            test_dataloader = DataLoader(
                test_dataset, batch_size=bs, collate_fn=collate_fn,
                shuffle=shuffle, num_workers=num_workers)
            dataloaders["val"] = test_dataloader
    return dataloaders
