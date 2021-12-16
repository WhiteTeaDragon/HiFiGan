import logging
import torchaudio
import torch
import librosa

from hifigan.datasets.utils import initialize_mel_spec
from hifigan.utils import ROOT_PATH

logger = logging.getLogger(__name__)


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, config_parser, data_dir=None, download="True"):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "lj"
            data_dir.mkdir(exist_ok=True, parents=True)
        super().__init__(root=data_dir, download=(download == "True"))
        self.config_parser = config_parser
        self.wave2spec = initialize_mel_spec(self.config_parser)

    def __getitem__(self, index: int):
        waveform, _, _, _ = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        melspec, _ = self.get_spectrogram(waveform)
        melspec_length = torch.tensor([melspec.shape[-1]]).int()

        return waveform, waveform_length, melspec, melspec_length

    def initialize_mel_spec(self):
        sr = self.config_parser["preprocessing"]["sr"]
        args = self.config_parser["preprocessing"]["spectrogram"]["args"]
        mel_basis = librosa.filters.mel(
            sr=sr,
            n_fft=args["n_fft"],
            n_mels=args["n_mels"],
            fmin=args["f_min"],
            fmax=args["f_max"]
        ).T
        wave2spec = self.config_parser.init_obj(
            self.config_parser["preprocessing"]["spectrogram"],
            torchaudio.transforms,
        )
        wave2spec.mel_scale.fb.copy_(torch.tensor(mel_basis))
        return wave2spec

    def get_spectrogram(self, audio_tensor_wave: torch.Tensor):
        sr = self.config_parser["preprocessing"]["sr"]
        with torch.no_grad():
            mel = self.wave2spec(audio_tensor_wave) \
                .clamp_(min=1e-5) \
                .log_()
        return mel, sr
