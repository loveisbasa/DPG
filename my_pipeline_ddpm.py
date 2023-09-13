from diffusers import DDPMPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor


from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch

class MyDDPMPipeline(DDPMPipeline):
    def __call__(
        self,
        measurement=None,
        mask=None,
        device=None,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        # image_length = measurement.shape[-1]
        # image_shape = (batch_size, 3, image_length,image_length)
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)


        for t in self.progress_bar(self.scheduler.timesteps):
            image = image.requires_grad_()
            # 1. predict noise model_output
            # time = torch.tensor([t] * image.shape[0], device=device)
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            out = self.scheduler.step(model_output, t, image, generator=generator, return_dict=True)
            image_now, pred_xstart = out['prev_sample'], out['pred_original_sample']

            #!!
            noise = torch.randn_like(measurement)
            noisy_measurement = self.scheduler.add_noise(measurement, noise = noise, timesteps=t)

            #!!
            image_now, distance = self.cond_fn(x_t=image_now,
                        measurement=measurement,
                        noisy_measurement=noisy_measurement,
                        x_prev=image,
                        x_0_hat=pred_xstart,
                        mask=mask
                        )
            #!!
            image = image_now.detach()


        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
