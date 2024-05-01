import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import random
import yaml
import cv2
import inflect
from torchvision import transforms
from PIL import Image
from dotmap import DotMap
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from tools.models.new_mod import Counter 
from tools.dataset.datasets import get_val_loader
from tools.util import save_density_map, get_model_dir
from tools.dataset.tokenizer import tokenize

class MainTransform(object):
    def __init__(self, img_size = 384):
        self.img_size = img_size
        self.img_trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
                        ])
    def __call__(self, img_path):
        img = Image.open(img_path)
        img.load()
        W, H = img.size
        new_H = 384
        new_W = 16 * int((W / H * 384) / 16)
        scale_factor_W = float(new_W) / W
        scale_factor_H = float(new_H) / H
        resized_img = transforms.Resize((new_H, new_W))(img)
        resized_img = self.img_trans(resized_img)

        return resized_img

config = 'config_files/FSC.yaml'
gpus = 0
enc = 'spt'
num_tokens = 10
patch_size = 16
prompt = 'plural'
ckpt_used = 'D:/CSAM/VLCounter/pretrain/182_best.pth'
exp = 1
with open(config, 'r') as f:
        config = yaml.safe_load(f)
args = DotMap(config)
args.config = config
args.gpus = gpus
args.enc = enc
args.num_tokens = num_tokens
args.patch_size = patch_size
args.prompt = prompt
args.EVALUATION.ckpt_used = ckpt_used
args.exp = exp

if args.TRAIN.manual_seed is not None:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(args.TRAIN.manual_seed)
    np.random.seed(args.TRAIN.manual_seed)
    torch.manual_seed(args.TRAIN.manual_seed)
    torch.cuda.manual_seed_all(args.TRAIN.manual_seed)
    random.seed(args.TRAIN.manual_seed)
model = Counter(args).cuda()
root_model = get_model_dir(args)
model.eval()

multi_plural_prompt_templates = ['a photo of a number of {}.',
                                'a photo of a number of small {}.',
                                'a photo of a number of medium {}.',
                                'a photo of a number of large {}.',
                                'there are a photo of a number of {}.',
                                'there are a photo of a number of small {}.',
                                'there are a photo of a number of medium {}.',
                                'there are a photo of a number of large {}.',
                                'a number of {} in the scene.',
                                'a photo of a number of {} in the scene.',
                                'there are a number of {} in the scene.',
                            ]
single_plural_prompt_templates = ['A photo of a {}.',
                                'A photo of a small {}.',
                                'A photo of a medium {}.',
                                'A photo of a large {}.',
                                'This is a photo of a {}.',
                                'This is a photo of a small {}.',
                                'This is a photo of a medium {}.',
                                'This is a photo of a large {}.',
                                'A {} in the scene.',
                                'A photo of a {} in the scene.',
                                'There is a {} in the scene.',
                                'There is the {} in the scene.',
                                'This is a {} in the scene.',
                                'This is the {} in the scene.',
                                'This is one {} in the scene.',
                            ]
class_chosen = 'markers'
img_path = r'C:\Users\Admin\Documents\LearningToCountEverything-master\data\images_384_VarV2\1123.jpg'

engine = inflect.engine()
if args.prompt == "plural":
    text = [template.format(engine.plural(class_chosen)) if engine.singular_noun(class_chosen) == False else template.format(class_chosen) for template in multi_plural_prompt_templates]
elif args.prompt == "single":
    text = [template.format(class_chosen) if engine.singular_noun(class_chosen) == False else template.format(engine.plural(class_chosen)) for template in single_plural_prompt_templates]
m_t = MainTransform()

tokenized_text = tokenize(text).unsqueeze(0)
query_img = m_t(img_path).unsqueeze(0)

query_img, tokenized_text = query_img.cuda(), tokenized_text.cuda()
_, _, h, w = query_img.shape
density_map = torch.zeros([h, w])
density_map = density_map.cuda()
attn_map = torch.zeros([h, w])
attn_map = attn_map.cuda()
start = 0
prev = -1
with torch.no_grad():
    while start + 383 < w:
        output, attn, _ = model(query_img[:, :, :, start:start + 384], tokenized_text)
        output = output.squeeze(0).squeeze(0)
        # attn = attn.squeeze(0).squeeze(0)
        b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
        d1 = b1(output[:, 0:prev - start + 1])
        # a1 = b1(attn[:, 0:prev - start + 1])
        b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
        d2 = b2(output[:, prev - start + 1:384])
        # a2 = b2(attn[:, prev - start + 1:384])

        b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
        density_map_l = b3(density_map[:, 0:start])
        density_map_m = b1(density_map[:, start:prev + 1])
        # attn_map_l = b3(attn_map[:, 0:start])
        # attn_map_m = b1(attn_map[:, start:prev + 1])
        b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
        density_map_r = b4(density_map[:, prev + 1:w])
        # attn_map_r = b4(attn_map[:, prev + 1:w])

        density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2
        # attn_map = attn_map_l + attn_map_r + attn_map_m / 2 + a1 / 2 + a2


        prev = start + 383
        start = start + 128
        if start + 383 >= w:
            if start == w - 384 + 128:
                break
            else:
                start = w - 384

density_map /= 60.
print(density_map.sum().item())