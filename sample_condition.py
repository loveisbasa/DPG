from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
# from guided_diffusion.gaussian_diffusion import create_diffusion
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator, clear
from util.logger import get_logger

# tensorboard for debugging
from torch.utils.tensorboard import SummaryWriter

import time
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--record', action="store_true")
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--start_n', type=int, default=0)
    args = parser.parse_args()

    setup_seed(0)

    # logger
    logger = get_logger()
    if args.record:
        writer = SummaryWriter()
    else:
        writer = None
    if args.record:
        record_loss = np.zeros((args.n, 1000))

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")

    # Load diffusion sampler
    # diffusion = create_diffusion(**diffusion_config)
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn, writer=writer)
    # sample_fn = (diffusion.p_sample_loop if diffusion_config['sampler']=="ddpm" else diffusion.ddim_sample_loop)
    # sample_fn = partial(sample_fn, model, (1, 3, model_config['image_size'], model_config['image_size']), clip_denoised=diffusion_config['clip_denoised'])

    # Working directory
    if measure_config['operator']['name'] == 'inpainting':
        task_name = 'inpainting_' + measure_config['mask_opt']['mask_type']
    elif measure_config['operator']['name'] == 'super_resolution':
        task_name = 'sr_' + str(measure_config['operator']['scale_factor'])
    else:
        task_name = measure_config['operator']['name']
    # import pdb
    # pdb.set_trace()
    out_path = os.path.join(args.save_dir, task_name+'_'+task_config['data']['name']+'_'+task_config['conditioning']['method'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
    if measure_config['operator']['name'] == 'inpainting':
        os.makedirs(os.path.join(out_path,'mask'), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    if task_config['data']['name'] == 'ffhq' or task_config['data']['name'] == 'exp':
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif task_config['data']['name'] == 'imagenet' or task_config['data']['name'] == 'uscsipi':
        def rescale(image): return (image - 0.5) * 2.
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(rescale)])
    elif task_config['data']['name'] == 'lsun':
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        raise ValueError(f"Unknown dataset {data_config['data']}")
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )

    # timer
    total_time = 0.
    psnr_func = PeakSignalNoiseRatio(data_range=2.).to(device)
    lpips_func = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='sum').to(device)


    # Do Inference
    for i, ref_img in enumerate(loader):
        if i < args.start_n: continue
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)

        else:
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)

        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        t_start = time.time()
        sample, outs = sample_fn(x_start=x_start, measurement=y_n, record=args.record, save_root=out_path, num_run=i)

        if args.record:
            record_loss[i] = np.array(outs['reconloss'])


        total_time += (time.time() - t_start)

        # lpips
        with torch.no_grad():
            psnr = psnr_func(ref_img, sample).cpu().detach().item()
            lpips_score = lpips_func(torch.clamp(ref_img, min=-1, max=1), torch.clamp(sample, min=-1, max=1)).cpu().detach().item()
        print('LPIPS: ', lpips_score,
            'Reconstruction MSE: ', psnr)

        if measure_config['operator']['name'] == 'color':
            plt.imsave(os.path.join(out_path, 'input', fname), clear(y_n), cmap='gray')
        else:
            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))
        # if measure_config['operator']['name'] == 'inpainting':
        #     plt.imsave(os.path.join(out_path,'mask', fname), clear_color(y))
        if writer is not None: writer.close()

        if args.n != -1 and i == (args.n - 1):
            break
    output = {'per_img_timer': total_time / (i + 1)}
    analysis_pt = out_path + '/analysis.pt'
    if args.record:
        np.save(cond_config['method'], record_loss)
    print(analysis_pt)
    torch.save(output, analysis_pt)



if __name__ == '__main__':
    main()
