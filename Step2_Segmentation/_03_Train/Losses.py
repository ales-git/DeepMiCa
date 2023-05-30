import torch
import torch.nn as nn


class topk_CE(nn.Module):
    def __init__(self, reduction='none'):
        super(topk_CE, self).__init__()
        self.reduction = reduction
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        loss = self.loss(input, target)
        batch_size = len(target)
        loss_final = []

        for i in range(batch_size):
            loss_w = loss[i, :, :, :][[target[i, :, :, :] == 1]]
            n_white = len(loss_w)

            loss_b = loss[i, :, :, :][[target[i, :, :, :] == 0]]
            ordered_loss_b, _ = torch.sort(loss_b, 0, descending=True)

            loss_final.append(loss_w)
            loss_final.append(ordered_loss_b[:3 * n_white])

        return torch.mean(torch.cat(loss_final))
