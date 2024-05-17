!pip install -r requirements.txt
!pip install hub
!pip install dotmap
!pip install inflect
!pip install ftfy
!pip install timm
import os
import torch
import torchvision
import argparse
import json
import numpy as np
import os
import copy
import time
import cv2
from tqdm import tqdm
from os.path import exists,join
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import clip
from shi_segment_anything import sam_model_registry, SamPredictor
# from shi_segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from shi_segment_anything.automatic_mask_generator_text import SamAutomaticMaskGenerator
from utils import *
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
from tools.models.VLCounter import Counter
from tools.dataset.datasets import get_val_loader
from tools.util import save_density_map, get_model_dir
from tools.dataset.tokenizer import tokenize

config = 'config_files/FSC.yaml'
gpus = 0
enc = 'spt'
num_tokens = 10
patch_size = 16
prompt = 'plural'
ckpt_used = './pretrain/182_best.pth'
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
if args.EVALUATION.ckpt_used is not None:
    filepath = args.EVALUATION.ckpt_used
    assert os.path.isfile(filepath), filepath
    print("=> loading model weight '{}'".format(filepath),flush=True)
    checkpoint = torch.load(filepath)
    # for key, value in checkpoint.items() :
    #     print(key)
    # print(checkpoint['epoch'])
    # print(checkpoint['best_mae'])
    # print(checkpoint['optimizer']['param_groups'][0]['lr'])
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model weight '{}'".format(filepath),flush=True)
else:
    print("=> Not loading anything",flush=True)

model.eval()

device = 'cuda:0'
sam_checkpoint = "./pretrain/sam_vit_b_01ec64.pth"
model_type = "vit_b"
clip_model, _ = clip.load("CS-ViT-B/16", device=device)
clip_model.eval()
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
print('loaded model')

mask_generator = SamAutomaticMaskGenerator(model=sam)

def box_from_mask(mask):
    bboxes = []
    for segmentation in mask:
        x_min = segmentation['bbox'][0]
        x_max = segmentation['bbox'][0] + segmentation['bbox'][2]
        y_min = segmentation['bbox'][1]
        y_max = segmentation['bbox'][1] + segmentation['bbox'][3]
        bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes

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

def check_bb(img_path, class_chosen):
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
            attn = attn.squeeze(0).squeeze(0)
            b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
            d1 = b1(output[:, 0:prev - start + 1])
            a1 = b1(attn[:, 0:prev - start + 1])
            b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
            d2 = b2(output[:, prev - start + 1:384])
            a2 = b2(attn[:, prev - start + 1:384])

            b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
            density_map_l = b3(density_map[:, 0:start])
            density_map_m = b1(density_map[:, start:prev + 1])
            attn_map_l = b3(attn_map[:, 0:start])
            attn_map_m = b1(attn_map[:, start:prev + 1])
            b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
            density_map_r = b4(density_map[:, prev + 1:w])
            attn_map_r = b4(attn_map[:, prev + 1:w])

            density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2
            attn_map = attn_map_l + attn_map_r + attn_map_m / 2 + a1 / 2 + a2


            prev = start + 383
            start = start + 128
            if start + 383 >= w:
                if start == w - 384 + 128:
                    break
                else:
                    start = w - 384

    density_map /= 60.
    print(density_map.sum().item())
    #re = format_for_plotting(density_map)
    # print(np.array(attn_map.cpu().shape))
    # print(np.array(density_map.cpu().shape))
    # print(np.unique(np.array(density_map.cpu())))
    map = np.array(attn_map.unsqueeze(-1).cpu())
    _, similarity_mask = cv2.threshold(map, np.max(map)/1.3, 1, cv2.THRESH_BINARY)

    # labeled_img, num = label(similarity_mask, background=0, return_num=True)
    # cv2.imshow('Result', np.array(similarity_mask))
    # # # cv2.imwrite('result.jpg', np.array(density_map.unsqueeze(-1).cpu()*255))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(similarity_mask.shape)
    image = cv2.imread(img_path)
    image = cv2.resize(image, (similarity_mask.shape[1], similarity_mask.shape[0]))
    # print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cls_name = class_chosen
    # input_prompt= select_max_region(similarity_mask)
    contours, _ = cv2.findContours(similarity_mask.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # print(input_prompt[0])
    input_prompt = []
    annot = image.copy()
    for cntr in contours:
        rect = cv2.boundingRect(cntr)
        x,y,w,h = rect
        cv2.rectangle(annot, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 0, 255), 2)
        # print("x,y,w,h:",x,y,w,h)
        input_prompt.append((x, y, x + w, y+h))
    # cv2.imshow("image", annot)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    filename = os.path.basename(img_path)
    image_number = filename.split('.')[0]
    print('get input_prompt')
    masks = mask_generator.generate(image, input_prompt)
    print(f'detected {os.path.basename(img_path)}')
    # print(masks[0])
    print(len(masks))
    pred = box_from_mask(masks)
    fin = image.copy()
    bb[os.path.basename(img_path)] = pred
    with open('data.json', 'w') as json_file:
      json.dump(bb, json_file)
    for box in pred:
        x,y,w,h = box
        cv2.rectangle(fin, (int(x), int(y)), (int(w), int(h)), (0, 0, 255), 2)
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    # Plot the first image
    axes[0, 0].imshow(image)
    axes[0, 0].axis('off')  # Turn off axis for clarity
    # Plot the second image
    axes[0, 1].imshow(annot)
    axes[0, 1].axis('off')  # Turn off axis for clarity
    # Plot the third image
    axes[1, 0].imshow(fin)
    axes[1, 0].axis('off')  # Turn off axis for clarity
    # Show the plot
    plt.savefig(f'./Result/{image_number}_annot_{len(pred)}.png')
    # cv2.imshow('result', cv2.imread(f'./Result/{image_number}_annot.png'))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return similarity_mask.shape[1], similarity_mask.shape[0], input_prompt

class_chosen = 'sea shells'
img_path = '/content/drive/MyDrive/DL/FSC147_384_V2/images_384_VarV2/2.jpg'
image_dir = '/content/drive/MyDrive/DL/FSC147_384_V2/images_384_VarV2/'
bb = dict()

# a, b, input_prompt = check_bb(img_path, class_chosen)
# print(bb)

with open(r'/content/drive/MyDrive/DL/FSC147_384_V2/ImageClasses_FSC147.txt') as f:
    class_lines = f.readlines()

class_dict = {}
for cline in class_lines:
    strings = cline.strip().split('\t')
    class_dict[strings[0]] = strings[1]

with open(r'/content/drive/MyDrive/DL/FSC147_384_V2/Train_Test_Val_FSC_147.json') as f:
    data_split = json.load(f)
im_ids_train = data_split['train']
im_ids_test = data_split['test']
im_ids_val = data_split['val']

for im_id in im_ids_test:
    cls_name = class_dict[im_id]
    img_path = image_dir + im_id
    check_bb(img_path, cls_name)