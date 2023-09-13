CUDA_VISIBLE_DEVICES=0 python3 infer_ddpm.py \
    --task_config=configs/inpainting_config_ddpm.yaml \
    --save_dir ./results_ddpm \
    --scheduler_type DDPM --num_inference_steps 500


# CUDA_VISIBLE_DEVICES=0 python3 infer_ddpm.py \
#     --task_config=configs/inpainting_config_ddpm.yaml \
#     --save_dir ./results_ddpm \
#     --scheduler_type DDIM --num_inference_steps 50


