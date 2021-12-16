import torch


class HiFiLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> dict:
        model_output = kwargs["output_melspec"]
        target = kwargs["melspec"]
        loss = torch.nn.functional.l1_loss(model_output, target)
        return loss

    def disc_forward(self, result, labels):
        return torch.nn.functional.binary_cross_entropy(result, labels)
