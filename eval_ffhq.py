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

    args.metrics = {'lpips': True,
            'individual-lpips': True,
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

    real_tensor, recon_tensor = torch.stack([T.ToTensor()(x) for x in img_real]), torch.stack([T.ToTensor()(x) for x in img_recon])

    output = torch.load(out_path) if os.path.exists(out_path) else {}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.metrics['individual-lpips']:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True, reduction='sum')
        with open(args.folder+'/lpips.txt', 'w+') as lpips_txt:
            for idx in range(args.num_pic):
                score = lpips(real_tensor[idx].unsqueeze(0), recon_tensor[idx].unsqueeze(0)).cpu().detach().item()
                lpips_txt.write("{:n}: {:.4f}\n".format(idx, score))


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
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        metric = StructuralSimilarityIndexMeasure(data_range=1.0, gaussian_kernel = False)
        output['ssim'] = metric(real, fake)



    # compute inception scores
    if args.metrics['fid']:
        from torchmetrics.image.fid import FrechetInceptionDistance

        real_img, fake_img = torch.stack([resize(299)(i_real) for i_real in img_real]), torch.stack([resize(299)(i_recon) for  i_recon in img_recon])

        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        for real_batch in real_img.chunk(10): fid.update(real_batch.to(device), real=True)
        for fake_batch in fake_img.chunk(10): fid.update(fake_batch.to(device), real=False)
        output['fid'] = fid.compute()

        from pytorch_fid.inception import InceptionV3
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(device)
        fake_img = torch.stack([resize(299)(i_recon) for i_recon in img_recon])
        real_img = torch.stack([resize(299)(i_recon) for i_recon in img_real])
        real_pred, fake_pred = np.empty((real_img.shape[0], 2048)), np.empty((fake_img.shape[0], 2048))
        start_idx = 0
        for real_batch, fake_batch in zip(real_img.chunk(32), fake_img.chunk(32)):
            with torch.no_grad():
                real_stats, fake_stats = model(real_batch.to(device))[0], model(fake_batch.to(device))[0]
            real_stats, fake_stats = real_stats.squeeze(3).squeeze(2).cpu().numpy(), fake_stats.squeeze(3).squeeze(2).cpu().numpy()
            real_pred[start_idx:start_idx + real_stats.shape[0]], fake_pred[start_idx:start_idx + fake_stats.shape[0]] = real_stats, fake_stats
            start_idx += real_stats.shape[0]
        mu, mu_ref = np.mean(real_pred, axis=0), np.mean(fake_pred, axis=0)
        sigma, sigma_ref = np.cov(real_pred, rowvar=False), np.cov(fake_pred, rowvar=False)
        # # pred_arr = np.empty((fake_img.shape[0], 2048))
        # # start_idx = 0
        # # for batch in fake_img.chunk(10):
        # #     batch = batch.to(device)
        # #     with torch.no_grad():
        # #         pred = model(batch)[0]
        # #     pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        # #     pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        # #     start_idx += pred.shape[0]

        # import pdb
        # pdb.set_trace()

        # fake_img = torch.stack([resize(299)(i_recon) for  i_recon in img_recon])
        # pred_arr = np.empty((fake_img.shape[0], 2048))
        # start_idx = 0
        # for real_batc, fake_batch in zip(real_tensor.chunk(32), recon_tensor.chunk(32)):
        #     batch = batch.to(device)
        #     with torch.no_grad():
        #         pred = model(batch)[0]
        #     pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        #     pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        #     start_idx += pred.shape[0]

        # mu = np.mean(pred, axis=0)
        # sigma = np.cov(pred, rowvar=False)


        # mu_ref = training_stats['mu']
        # sigma_ref = training_stats['sigma']






        # from torchmetrics.image.fid import FrechetInceptionDistance
        # # real_img_folder = Path("/home/tanghaoyue13/dataset/ffhq256")
        # fpaths = sorted(glob("/home/tanghaoyue13/dataset/ffhq256" + '/**/*.png', recursive=True))
        # img_validation_set = [load(x) for x in fpaths]
        # fake_img = torch.stack([resize(299)(i_recon) for i_recon in img_recon])
        # real_img = torch.stack([resize(299)(i_recon) for i_recon in img_validation_set])

        # # real_img, fake_img = torch.stack([resize(299)(i_real) for i_real in img_real]), torch.stack([resize(299)(i_recon) for  i_recon in img_recon])

        # fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        # batch_idx = 0
        # for real_batch in real_img.chunk(10):
        #     print(batch_idx)
        #     batch_idx += 1
        #     fid.update(real_batch.to(device), real=True)
        # for fake_batch in fake_img.chunk(10): fid.update(fake_batch.to(device), real=False)
        # output['fid'] = fid.compute()
        # # from cleanfid import fid
        # # # output['fid'] = fid.compute_fid(args.folder + '/recon', dataset_name="FFHQ", dataset_res=256, dataset_split="train256")




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
