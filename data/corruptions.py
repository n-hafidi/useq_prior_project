import torch
import random


def hole_mask(img,size):

    mask = torch.ones_like(img)

    c = img.shape[-1]//2

    mask[:,:,c-size//2:c+size//2,c-size//2:c+size//2] = 0

    return mask


def random_mask(img,rate):

    return (torch.rand_like(img) > rate).float()


def text_mask(img,num_lines,thickness):

    mask = torch.ones_like(img)

    h = img.shape[-1]

    for _ in range(num_lines):

        x = random.randint(0,h-thickness)

        mask[:,:,:,x:x+thickness] = 0

        y = random.randint(0,h-thickness)

        mask[:,:,y:y+thickness,:] = 0

    return mask