import torch
from losses.losses import total_loss


class Trainer:

    def __init__(self, model, optimizer, cfg):

        self.model = model
        self.opt = optimizer
        self.cfg = cfg
        self.loss_curve = []

    #def train(self, z, corrupted, mask):
    def train(self, z, corrupted, mask, clean):

        iters = self.cfg["training"]["iterations"]

        for i in range(iters):

            self.opt.zero_grad()

            sigma = 0.01 * (1 - i / iters)
            noise = torch.randn_like(z) * sigma

            pred = self.model(z + noise, mask)

            #loss = total_loss(
            #    pred,
            #    corrupted,
            #    mask,
            #    i,
            #    iters,
            #    self.cfg
            #)
            
            loss = total_loss(
                pred,
                clean,
                mask,
                i,
                iters,
                self.cfg
            )

            loss.backward()
            self.opt.step()

            self.loss_curve.append(loss.item())

            if i % 50 == 0:
                print(f"Iter {i}/{iters} Loss: {loss.item():.6f}")

        return pred.detach()