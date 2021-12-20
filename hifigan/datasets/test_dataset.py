import logging
import os

import torch
import torchaudio
from torch.utils.data import Dataset

from hifigan.utils import ROOT_PATH

logger = logging.getLogger(__name__)


class TestDataset(Dataset):
    def __init__(self, config_parser, segment_size, data_dir=None,
                 download="True"):
        super().__init__()
        self.config_parser = config_parser
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "test"
            data_dir.mkdir(exist_ok=True, parents=True)
        self.length = len(os.listdir(data_dir))
        self.data_dir = data_dir

    def __getitem__(self, index: int):
        waveform, _ = torchaudio.load(self.data_dir / f"{index}.wav")
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        return {"audio": waveform, "audio_length": waveform_length}

    def __len__(self):
        return self.length
