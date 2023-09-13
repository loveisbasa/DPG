from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger


from my_pipeline_ddpm import MyDDPMPipeline
from diffusers import DDIMScheduler, DDPMScheduler

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
    parser.add_argument('--scheduler_type', type=str, default='DDPM')
    parser.add_argument('--num_inference_steps', type=int, default=50)
    
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    task_config = load_yaml(args.task_config)
   
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."


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
       
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name']+"_"+args.scheduler_type+str(args.num_inference_steps))
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    #!!
    transform = transforms.Compose([transforms.Resize(data_config.pop("size")),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
        
    #!!   
    # # load an online model
    pipeline = MyDDPMPipeline.from_pretrained("google/ddpm-celebahq-256").to(device)
    if args.scheduler_type == "DDPM":
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    elif args.scheduler_type == "DDIM":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    else:
        exit("Add this new scheduler to code!")
    
    # # load local model if needed
    # model_config = load_yaml(args.model_config)
    # unet = create_model(**model_config)
    # unet = unet.to(device)
    # unet.eval()
    # if args.scheduler_type == "DDPM":
    #     scheduler = DDPMScheduler.from_config("./scheduler_config_ddpm.json")
    # elif args.scheduler_type == "DDIM":
    #     scheduler = DDIMScheduler.from_config("./scheduler_config_ddpm.json")
    # else:
    #     exit("Add this new scheduler to code!")
    # pipeline = MyDDPMPipeline(unet=unet,scheduler=scheduler)
    
    
    
    pipeline.cond_fn = cond_method.conditioning
    pipeline.operator_name = measure_config['operator']['name']

    
    # Do Inference
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)   #[1, 3, 256, 256]

        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask = mask)
            y_n = noiser(y)   

        else: 
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)   #[1, 3, 64, 64]
            y_n = y_n.detach()
            mask = None
        # Sampling
        generator = torch.Generator(device=device).manual_seed(42)
        sample = pipeline( 
                            measurement = y_n,
                            mask = mask,
                            num_inference_steps = args.num_inference_steps,
                            generator=generator,
                            device = device
                        ).images[0]

        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        sample.save(os.path.join(out_path, 'recon', fname))
        exit()

if __name__ == '__main__':
    main()