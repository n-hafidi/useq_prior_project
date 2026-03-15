import torch
from losses.losses import total_loss


class Trainer:

    def __init__(self, model, lr, cfg):
        
        #added for accelaration
        #self.scaler = torch.cuda.amp.GradScaler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        self.model = model
        self.cfg = cfg

        self.opt = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )

    # def train(self, z, corrupted, mask, step):

        # self.opt.zero_grad()

        # iters = self.cfg["training"]["iterations"]

        # sigma = 0.03 * (1 - step / iters)

        # noise = torch.randn_like(z) * sigma

        # pred = self.model(z + noise, mask)

        # loss = total_loss(
            # pred,
            # corrupted,
            # mask,
            # step,
            # iters,
            # self.cfg
        # )

        # loss.backward()

        # self.opt.step()

        # return pred, loss.item()
        
    # Accelerated function
    def train(self, z, corrupted, mask, step):

        self.opt.zero_grad()

        iters = self.cfg["training"]["iterations"]

        sigma = 0.03 * (1 - step / iters)

        noise = torch.randn_like(z) * sigma
        
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        #with torch.cuda.amp.autocast():
        with torch.autocast(device_type=device_type):

            pred = self.model(z + noise, mask)

            loss = total_loss(
                pred,
                corrupted,
                mask,
                step,
                iters,
                self.cfg
            )

        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()

        return pred, loss.item()