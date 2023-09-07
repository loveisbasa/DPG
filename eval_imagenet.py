import torch
import argparse
from pathlib import Path
import torchvision.transforms as T
import json
from PIL import Image
import lpips
import numpy as np
import os
import yaml
from evalutils.io import load, dump
from evalutils.torch_utils import *
from torchvision.models import resnet50
from torchvision import transforms
import torchvision
from data.dataloader import get_dataset, get_dataloader
from torchvision.datasets import VisionDataset
from glob import glob
import math
import numpy

resize = lambda n:T.Compose([
    T.Resize(n),
    T.CenterCrop(n),
    T.ToTensor(),
])

torch.set_grad_enabled(False)

preprocess = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms
        self.label_file = open("/home/tanghaoyue13/dataset/dataset/ILSVRC2012_validation_ground_truth.txt", 'r')
        contents = label_file.read()
        labels = contents.split('\n')[0:50000]
        self.labels = [int(label) for label in labels[:len(self.label_file)]]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, self.labels[i]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, default="outputs/imagenet_edit_samples0719_1", help="folder for images")
    parser.add_argument("-d", "--device", type=str, default='cuda:0', help="cuda id")
    parser.add_argument("-m", "--metric", type=str, default="", help="which metric to run (by default: all)")
    parser.add_argument("-b", "--block", type=int, default=3, help="Inception block")
    parser.add_argument("-n", "--num_pic", type=int, default=5)

    args, unknown = parser.parse_known_args()
    args.ori_folder = Path(args.folder+'/label')
    args.edit_folder = Path(args.folder+'/recon')

    args.metrics = {'deit': True,
                    'lpips': True,
                    'psnr': True,
                    'fid': True,
                    'ssim': True}
    if args.metric:
        args.metrics = {k: (k == args.metric) for k in args.metrics}

    return args

def evaluate(args):

    label_folder, recon_folder = Path(args.folder + '/label'), Path(args.folder + '/recon')
    out_path = args.folder + '/analysis.pt'

    # real_img, edit_img
    label_paths, recon_paths = [x for x in label_folder.iterdir()], [x for x in recon_folder.iterdir()]
    if args.num_pic > 0:
        label_paths, recon_paths = label_paths[:args.num_pic], recon_paths[:args.num_pic]
    img_real, img_recon = [load(x) for x in label_paths], [load(x) for x in recon_paths]

    # # get labels for accuracy calculation
    # label_file = open("/home/tanghaoyue13/dataset/dataset/val.txt", 'r')
    # contents = label_file.read()
    # labels = contents.split('\n')[0:50000]
    # labels = [int(tmp.split(' ')[-1]) for tmp in labels]
    labels = [idx for idx in range(1000)]
    labels = torch.tensor(labels[0:args.num_pic])
    real_tensor, recon_tensor = torch.stack([T.ToTensor()(x) for x in img_real]), torch.stack([T.ToTensor()(x) for x in img_recon])


    output = {}

    if args.metrics['deit']:
        imgt_classif = torch.stack([preprocess(im) for im in img_recon])
        model = resnet50(pretrained=True)

        # in_norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        # model = torch.hub.load('facebookresearch/deit:main',
        #                        'deit_base_distilled_patch16_384',
        #                        pretrained=True,
        #                       verbose=False).to(args.device)
        model.training = False

        preds = model.batch_forward(imgt_classif, batch_size=32)
        pred_idx = torch.argmax(preds, dim=-1)
        acc = (pred_idx == labels).float().mean().item()
        output['Accuracy'] = 100*acc



    # compute LPIPS w.r.t. input images
    if args.metrics['lpips']:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True, reduction='sum')
        lpips_score = 0
        num_imgs = 0
        for real_batch, fake_batch in zip(real_tensor.chunk(32), recon_tensor.chunk(32)):
            lpips_score += lpips(real_batch, fake_batch).cpu().detach().item()
            num_imgs += real_batch.shape[0]
        output['Mean LPIPS distance'] = lpips_score / num_imgs

    if args.metrics['psnr'] or args.metrics['ssim']:
        # real, fake = torch.stack([torch.from_numpy(numpy.array(i_real)) for i_real in img_real]), torch.stack([torch.from_numpy(numpy.array(i_recon)) for  i_recon in img_recon])
        real, fake = torch.stack([T.ToTensor()(i_real) for i_real in img_real]), torch.stack([T.ToTensor()(i_recon) for  i_recon in img_recon])


    if args.metrics['psnr']:
        from torchmetrics.image import PeakSignalNoiseRatio
        metric = PeakSignalNoiseRatio()
        output['psnr'] = metric(real, fake)

    if args.metrics['ssim']:
        from skimage.metrics import structural_similarity as ssim
        ssim_list = [ssim(numpy.asfarray(real), numpy.asfarray(recon), data_range=255., multichannel=True, gaussian_weights=False, sigma=1.5, channel_axis=2)
                    for real, recon in zip(img_real, img_recon)]
        output['ssim'] = torch.tensor(ssim_list).mean().item()



    # compute inception scores
    if args.metrics['fid']:
        from torchmetrics.image.fid import FrechetInceptionDistance

        real_img, fake_img = torch.stack([resize(299)(i_real) for i_real in img_real]), torch.stack([resize(299)(i_recon) for  i_recon in img_recon])

        fid = FrechetInceptionDistance(feature=2048, normalize=True)
        for real_batch in real_img.chunk(10): fid.update(real_batch, real=True)
        for fake_batch in fake_img.chunk(10): fid.update(fake_batch, real=False)
        output['fid'] = fid.compute()


    torch.save(output, out_path)
    return output


if __name__ == "__main__":
    args = get_args()
    print(f'Starting evaluation of folder {str(args.folder)}')
    output = evaluate(args)
    for k, v in output.items():
        if isinstance(v, float) or isinstance(v, int):
            print(f'{k}: {v:.3f}')
        else:
            print(f'{k}: {v}')
