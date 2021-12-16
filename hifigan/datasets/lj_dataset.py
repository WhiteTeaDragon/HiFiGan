import logging
import random

import torchaudio
import torch

from hifigan.datasets.utils import initialize_mel_spec
from hifigan.utils import ROOT_PATH

logger = logging.getLogger(__name__)


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, config_parser, segment_size, data_dir=None,
                 download="True"):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "lj"
            data_dir.mkdir(exist_ok=True, parents=True)
        super().__init__(root=data_dir, download=(download == "True"))
        self.config_parser = config_parser
        self.wave2spec = initialize_mel_spec(self.config_parser)
        self.segment_size = segment_size

    def __getitem__(self, index: int):
        waveform, _, _, _ = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()
        if self.segment_size is not None:
            if waveform_length > self.segment_size:
                max_audio_start = waveform_length - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                waveform = waveform[:, audio_start:audio_start +
                                                   self.segment_size]
            else:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.segment_size - waveform_length),
                    'constant')

        melspec, _ = self.get_spectrogram(waveform)
        melspec_length = torch.tensor([melspec.shape[-1]]).int()

        return waveform, waveform_length, melspec, melspec_length

    def get_spectrogram(self, audio_tensor_wave: torch.Tensor):
        sr = self.config_parser["preprocessing"]["sr"]
        with torch.no_grad():
            mel = self.wave2spec(audio_tensor_wave) \
                .clamp_(min=1e-5) \
                .log_()
        return mel, sr
