import torch
from losses.losses import total_loss


class Trainer:

    def __init__(self, model, lr, cfg):

        self.model = model
        self.cfg = cfg

        self.opt = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )

    def train(self, z, corrupted, mask, step):

        self.opt.zero_grad()

        iters = self.cfg["training"]["iterations"]

        sigma = 0.03 * (1 - step / iters)

        noise = torch.randn_like(z) * sigma

        pred = self.model(z + noise, mask)

        loss = total_loss(
            pred,
            corrupted,
            mask,
            step,
            iters,
            self.cfg
        )

        loss.backward()

        self.opt.step()

        return pred, loss.item()