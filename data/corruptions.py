import torch
import random

import torch
import numpy as np
import cv2


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
    
    
    



def text_words_mask(img, num_words=5, thickness=2):

    """
    Ajoute des mots aléatoires sur l'image et crée le mask correspondant.
    num_words contrôle la quantité de texte.
    """

    b, c, h, w = img.shape

    mask = np.ones((h, w), dtype=np.float32)

    vocabulary = [
        "AI", "DATA", "MODEL", "TEXT", "TEST",
        "VISION", "IMAGE", "MASK", "RESTORE", "DIP"
    ]

    for _ in range(num_words):

        word = np.random.choice(vocabulary)

        x = np.random.randint(0, w - 80)
        y = np.random.randint(20, h - 10)

        font_scale = np.random.uniform(0.6, 1.4)

        cv2.putText(
            mask,
            word,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            0,
            thickness,
            cv2.LINE_AA
        )

    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)

    return mask.to(img.device)