from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def compute_psnr(x,y):

    return peak_signal_noise_ratio(x,y,data_range=1)


def compute_ssim(x,y):

    return structural_similarity(x,y,data_range=1)