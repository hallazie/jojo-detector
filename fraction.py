# coding:utf-8

from PIL import Image

import numpy as np
import torch

# img = np.array(Image.open('data/th.jpg'))
# print(img.shape)

# exp = np.array([[[img[(x-1)//2, (h-1)//2, c] if x%2==0 else 0 for x in range(img.shape[-3]*2+1)] if h%2==0 else [0 for x in range(img.shape[-3]*2+1)] for h in range(img.shape[-2]*2+1)] for c in range(img.shape[-1])]).transpose()
# print(exp.shape)

# Image.fromarray(exp.astype('uint8')).save('data/th-exp.jpg')
