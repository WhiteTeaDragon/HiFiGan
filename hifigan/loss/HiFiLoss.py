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
        loss = 0
        for layer in target_res:
            loss += torch.mean((1 - layer) ** 2)
        return loss

    def disc_forward(self, target_res, model_res):
        loss = 0
        for layer1, layer2 in zip(target_res, model_res):
            loss += torch.mean((1 - layer1) ** 2) + torch.mean(layer2 ** 2)
        return loss

    def feature_loss(self, target_features, model_features):
        loss = 0
        for disc1, disc2 in zip(target_features, model_features):
            for layer1, layer2 in zip(disc1, disc2):
                loss += torch.mean(torch.abs(layer1 - layer2))
        return loss
