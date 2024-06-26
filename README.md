# Solving General Inverse Problems via Posterior Sampling: A Policy Gradient Viewpoint (AISTATS 2024)

<!-- See more results in the [project-page](https://jeongsol-kim.github.io/dps-project-page) -->

## Abstract
In this work, we solve image inverse problems (e.g., inpainting, super-resolution, deblurring) using a pretrained diffusion model. We improve the conditional score function estimated by DPS though the policy gradient method in reinforcement learning. To precisely estimate the guidance score function of the input image, we propose Diffusion Policy Gradient (DPG), a tractable computation method by viewing the intermediate noisy images as policies and the target image as the states selected by the policy. Experiments show that our method is robust to both Gaussian and Poisson noise degradation on multiple linear and non-linear inverse tasks, resulting into a higher image restoration quality on FFHQ, ImageNet and LSUN datasets.

![cover-img](./figures/cover.png)

This implementation is based on / inspired by:

https://github.com/DPS2022/diffusion-posterior-sampling (DPS)

## Prerequisites
- python 3.8

- pytorch 1.11.0

- CUDA 11.3.1


<br />

## Getting started

### 1) Clone the repository

```
git clone https://github.com/loveisbasa/DPG.git

cd DPG
```

<br />

### 2) Download pretrained checkpoint
The FFHQ pretrained model is from the DPS repo. The pretrained model for ImageNet (unconditional 256*256 ImageNet model) and LSUN (lsum_bedroom.pt) are from the guided diffusion paper from OpenAI https://github.com/openai/guided-diffusion. We store all the models in a folder.

From the [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing), download the checkpoint "ffhq_10m.pt" and paste it to ./models/
```
mkdir models
mv {DOWNLOAD_DIR}/ffqh_10m.pt ./models/
```
{DOWNLOAD_DIR} is the directory that you downloaded checkpoint to.

:speaker: Checkpoint for imagenet is uploaded.

<br />


### 3) Set environment
### [Option 1] Local environment setting

We use the external codes for motion-blurring and non-linear deblurring.

```
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse

git clone https://github.com/LeviBorodenko/motionblur motionblur
```

Install dependencies

```
conda create -n DPS python=3.8

conda activate DPS

pip install -r requirements.txt

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```


### 4) Inference

Currently DPG supports Inpainting, super-resolution, Gaussian and Motion debluring tasks, more task will be added soon.

To run DPG for super-resolution:


```
python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/ddim_config.yaml --task_config=configs/super_resolution_config.yaml
```

for inpainting

```
python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/ddim_config.yaml --task_config=configs/inpainting_config.yaml
```

:speaker: For imagenet, use configs/imagenet_model_config.yaml

<br />

## Possible task configurations

```
# Linear inverse problems
- configs/super_resolution_config.yaml
- configs/gaussian_deblur_config.yaml
- configs/motion_deblur_config.yaml
- configs/inpainting_config.yaml

# Non-linear inverse problems
- configs/nonlinear_deblur_config.yaml
```

### Structure of task configurations
You need to write your data directory at data.root. Default is ./data/samples which contains three sample images from FFHQ validation set.

```
conditioning:
    method: # check candidates in guided_diffusion/condition_methods.py
    params:
        scale: 0.5

data:
    name: ffhq
    root: ./data/samples/

measurement:
    operator:
        name: # check candidates in guided_diffusion/measurements.py

noise:
    name:   # gaussian or poisson
    sigma:  # if you use name: gaussian, set this.
    (rate:) # if you use name: poisson, set this.
```

## Citation
If you find our work interesting, please consider citing

```
@inproceedings{tang2024solving,
      title={Solving General Noisy Inverse Problem via Posterior Sampling: A Policy Gradient Viewpoint},
      author={Haoyue Tang and Tian Xie and Aosong Feng and Hanyu Wang and Chenyang Zhang and Yang Bai},
      booktitle = {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
      year={2024},
      series = 	 {Proceedings of Machine Learning Research},
      month = {2--4 May},
      eprint={2403.10585},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
