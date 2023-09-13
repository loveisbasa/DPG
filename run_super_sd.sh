# CUDA_VISIBLE_DEVICES=0 python3 infer_sd.py \
#     --task_config=configs/super_resolution_config_sd.yaml \
#     --save_dir ./results_sd \
#     --scheduler_type DDPM --num_inference_steps 500


# CUDA_VISIBLE_DEVICES=0 python3 infer_sd.py \
#     --task_config=configs/super_resolution_config_sd.yaml \
#     --save_dir ./results_sd \
#     --scheduler_type DDIM --num_inference_steps 50



#???
# CUDA_VISIBLE_DEVICES=0 python3 infer_sd.py \
#     --task_config=configs/super_resolution_config_sd_pg.yaml \
#     --save_dir ./results_sd_pg \
#     --scheduler_type DDIM --num_inference_steps 50