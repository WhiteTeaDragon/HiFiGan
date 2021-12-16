import random

import PIL
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor

from hifigan.base import BaseTrainer
from hifigan.datasets.utils import initialize_mel_spec
from hifigan.logger.utils import plot_spectrogram_to_buf
from hifigan.utils import inf_loop, MetricTracker, ROOT_PATH

from tqdm import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            generator,
            discriminator,
            criterion,
            metrics,
            optimizerG,
            optimizerD,
            config,
            device,
            data_loader,
            log_step,
            valid_data_loader=None,
            lr_schedulerG=None,
            lr_schedulerD=None,
            len_epoch=None,
            skip_oom=True,
            schedulerG_frequency_of_update=None,
            schedulerD_frequency_of_update=None
    ):
        super().__init__(generator, discriminator, criterion, metrics,
                         optimizerG, optimizerD, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.data_loader = data_loader
        self.log_step = log_step
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_schedulerG = lr_schedulerG
        self.schedulerG_frequency_of_update = schedulerG_frequency_of_update
        self.lr_schedulerD = lr_schedulerD
        self.schedulerD_frequency_of_update = schedulerD_frequency_of_update

        if discriminator is not None:
            self.train_metrics = MetricTracker(
                "generator loss", "generator reconstruction loss",
                "generator adversarial loss", "fake_D_loss", "real_D_loss",
                "generator grad norm", "discriminator grad norm",
                *[m.name for m in self.metrics]
            )
        else:
            self.train_metrics = MetricTracker(
                "generator loss",
                "generator grad norm",
                *[m.name for m in self.metrics]
            )
        self.valid_metrics = MetricTracker(
            "loss",
            *[m.name for m in self.metrics]
        )
        self.wave2spec = initialize_mel_spec(self.config, self.device)

    def get_spectrogram(self, audio_tensor_wave: torch.Tensor):
        sr = self.config["preprocessing"]["sr"]
        mel = self.wave2spec(audio_tensor_wave) \
            .clamp_(min=1e-5) \
            .log_()
        return mel, sr

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["audio", "audio_length", "melspec",
                               "melspec_length"]:
            if tensor_for_gpu in batch:
                batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip_G", None) is not None:
            clip_grad_norm_(
                self.generator.parameters(),
                self.config["trainer"]["grad_norm_clip_G"]
            )
        if self.config["trainer"].get("grad_norm_clip_D", None) is not None:
            clip_grad_norm_(
                self.discriminator.parameters(),
                self.config["trainer"]["grad_norm_clip_D"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        under_enumerate = tqdm(self.data_loader, desc="train",
                               total=self.len_epoch)
        for batch_idx, batch in enumerate(under_enumerate):
            try:
                batch = self.train_on_batch(
                    batch,
                    metrics=self.train_metrics
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.generator.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    if self.discriminator is not None:
                        for p in self.discriminator.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("generator grad norm",
                                      self.get_model_grad_norm(self.generator))
            if self.discriminator is not None:
                self.train_metrics.update("discriminator grad norm",
                                          self.get_model_grad_norm(
                                              self.discriminator))
            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Generator loss: {:.6f}".format(
                        epoch, self._progress(batch_idx),
                        batch["generator loss"]
                    )
                )
                self.writer.set_step((epoch - 1) * self.len_epoch +
                                     batch_idx)
                self.writer.add_scalar(
                    "generator learning rate", get_lr(self.optimizerG)
                )
                if self.discriminator is not None:
                    self.writer.add_scalar(
                        "discriminator learning rate", get_lr(
                            self.optimizerD)
                    )
                self._log_spectrogram(batch["melspec"],
                                      batch["output_melspec"])
                self._log_scalars(self.train_metrics)
                self._log_audio(batch["audio"], batch["output"], "train")
            if batch_idx >= self.len_epoch:
                break

        log = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_schedulerG is not None and \
                self.schedulerG_frequency_of_update == "epoch":
            if isinstance(self.lr_schedulerG,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                if not self.do_validation:
                    raise ValueError("Cannot use ReduceLROnPlateau if "
                                     "validation is off")
                self.lr_schedulerG.step(val_log["loss"])
            else:
                self.lr_schedulerG.step()
        if self.lr_schedulerD is not None and \
                self.schedulerD_frequency_of_update == "epoch":
            if isinstance(self.lr_schedulerD,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                if not self.do_validation:
                    raise ValueError("Cannot use ReduceLROnPlateau if "
                                     "validation is off")
                self.lr_schedulerD.step(val_log["loss"])
            else:
                self.lr_schedulerD.step()

        return log

    def train_on_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        batch["device"] = self.device
        if self.discriminator is not None:
            self.discriminator.train()
        self.generator.train()

        output_wav = self.generator(**batch)["output"]
        output_melspec, _ = self.get_spectrogram(output_wav)

        if self.discriminator is not None:
            ### Update discriminator
            self.discriminator.zero_grad(set_to_none=True)

            result = self.discriminator(batch["audio"], output_wav.detach())
            target_res, model_res, _, _ = result["mpd"]
            mpd_loss = self.criterion.disc_forward(target_res, model_res)
            mpd_loss.backward()

            target_res, model_res, _, _ = result["msd"]
            msd_loss = self.criterion.disc_forward(target_res, model_res)
            msd_loss.backward()

            self._clip_grad_norm()
            self.optimizerD.step()

        ### Update generator
        self.generator.zero_grad(set_to_none=True)
        batch["output"] = output_wav
        batch["output_melspec"] = output_melspec
        gen_reconstruction_loss = self.criterion(**batch)
        if self.discriminator is not None:
            result = self.discriminator(batch["audio"], output_wav)

            final_loss = gen_disc_loss + self.criterion.lam * \
                gen_reconstruction_loss
        else:
            final_loss = gen_reconstruction_loss
        final_loss.backward()
        self._clip_grad_norm()
        self.optimizerG.step()

        batch["generator loss"] = final_loss.item()

        if metrics is not None:
            metrics.update("generator loss", batch["generator loss"])
            if self.discriminator is not None:
                metrics.update("generator reconstruction loss",
                               gen_reconstruction_loss.item())
                metrics.update("generator adversarial loss",
                               gen_disc_loss.item())
                metrics.update("fake_D_loss", fake_loss.item())
                metrics.update("real_D_loss", real_loss.item())
            for met in self.metrics:
                metrics.update(met.name, met(**batch))
        return batch

    def valid_on_batch(self, batch, metrics):
        batch = self.move_batch_to_device(batch, self.device)
        batch["device"] = self.device
        self.generator.eval()
        if self.discriminator is not None:
            self.discriminator.eval()
        with torch.no_grad():
            outputs = self.generator(**batch)
            output_melspec, _ = self.get_spectrogram(outputs["output"])
            batch.update(outputs)
            batch["output_melspec"] = output_melspec
            loss = self.criterion(**batch)
            if metrics is not None:
                metrics.update("loss", loss.item())
                for met in self.metrics:
                    metrics.update(met.name, met(**batch))
        return batch

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.valid_metrics.reset()
        with torch.no_grad():
            for_range = tqdm(
                enumerate(self.valid_data_loader),
                desc="validation",
                total=len(self.valid_data_loader),
            )
            for batch_idx, batch in for_range:
                batch = self.valid_on_batch(
                    batch,
                    metrics=self.valid_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, "valid")
            self._log_scalars(self.valid_metrics)
            self._log_spectrogram(batch["melspec"], batch["output_melspec"])
            self._log_audio(batch["audio"], batch["output"], part="val")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def get_model_grad_norm(self, model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in
                 parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            if metric_name != "fid":
                self.writer.add_scalar(f"{metric_name}",
                                       metric_tracker.avg(metric_name))

    def _log_spectrogram(self, target_spectrograms, spectrogram_batch):
        log_index = torch.randint(low=0, high=len(target_spectrograms),
                                  size=(1,)).item()
        target_spec = target_spectrograms[log_index]
        image = PIL.Image.open(
            plot_spectrogram_to_buf(target_spec.detach().cpu()))
        self.writer.add_image("target spec", ToTensor()(image))
        output_spec = spectrogram_batch[log_index].squeeze(0)
        image = PIL.Image.open(
            plot_spectrogram_to_buf(output_spec.detach().cpu()))
        self.writer.add_image("output spec", ToTensor()(image))

    def _log_audio(self, input_batch, output_batch, part):
        index = random.choice(range(len(input_batch)))
        audio = input_batch[index]
        output_audio = output_batch[index]
        sample_rate = self.config["preprocessing"]["sr"]
        self.writer.add_audio("audio target" + part, audio, sample_rate)
        self.writer.add_audio("audio output" + part, output_audio, sample_rate)
