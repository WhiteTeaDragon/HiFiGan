import logging
from typing import List, Tuple, Dict
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)
MELSPEC_PAD_VALUE = -11.5129251


def collate_fn(instances: List[Tuple]) -> Dict:
    """
    Collate and pad fields in dataset items
    """
    input_data = list(zip(*instances))
    waveform, waveform_length = input_data

    waveform = pad_sequence([
        waveform_[0] for waveform_ in waveform
    ]).transpose(0, 1)
    waveform_length = torch.cat(waveform_length)

    return {"audio": waveform, "audio_length": waveform_length}
