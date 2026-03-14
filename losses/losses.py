import torch


def tv_loss(x):

    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))

    return tv_h + tv_w


def total_loss(pred, corrupted, mask, iteration, max_iter, cfg):

    # reconstruction (masked L1)
    rec_loss = torch.mean(torch.abs((pred - corrupted) * mask))

    # dynamic TV regularization
    tv_lambda = cfg["training"]["tv_lambda"]

    decay = 1 - iteration / max_iter

    tv = tv_lambda * decay * tv_loss(pred)

    return rec_loss + tv