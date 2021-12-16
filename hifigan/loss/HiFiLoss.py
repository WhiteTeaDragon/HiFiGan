import torch


class HiFiLoss(torch.nn.Module):
    def __init__(self, lam_fm=1, lam_mel=1):
        super().__init__()
        self.lam_fm = lam_fm
        self.lam_mel = lam_mel

    def forward(self, *args, **kwargs) -> dict:
        model_output = kwargs["output_melspec"]
        target = kwargs["melspec"]
        loss = torch.nn.functional.l1_loss(model_output, target)
        return loss

    def real_loss(self, target_res):
        return torch.sum(torch.mean((1 - target_res) ** 2, dim=0))

    def fake_loss(self, model_res):
        torch.sum(torch.mean(model_res ** 2, dim=0))

    def disc_forward(self, target_res, model_res):
        return self.real_loss(target_res) + self.fake_loss(model_res)

