#!/bin/bash

# 创建 logs 目录
logs_dir="logs"
mkdir -p "$logs_dir"

# 生成当前时间的字符串表示（年月日时分秒）
current_time=$(date +"%Y_%m_%d_%H_%M_%S")

seed=666
lr=0.001

resume=$1

# 创建一个以时间为名的子文件夹
subfolder="$logs_dir/finetune/$current_time"
mkdir -p "$subfolder"
echo "Generated subfolder for logging: $subfolder"

CUDA_VISIBLE_DEVICES=1 deepspeed --master_port 60000 main.py \
--deepspeed \
--deepspeed_config ds_fp16_config.json \
--multiprocessing_distributed \
--seed ${seed} \
--concat_train_val \
--lr ${lr} \
--epochs 90 \
--batch-size 16 \
--print-freq 100 \
--log_dir ${subfolder} \
$@ 2>&1 | tee "$subfolder/output.log" 


# ###### train resnet50
# bash finetune.sh -a resnet50 

# ###### resume searching/training model
# bash finetune.sh -a vit_b --resume /path/to/name.pt
