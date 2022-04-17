CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python3 -m torch.distributed.launch --nproc_per_node=8 --master_addr 127.0.0.1 --master_port 13579 train_mdeq_imagenet.py \
                --data "/path/to/data_rec" \
                --save-path "./log_dir" \
                --workers "8" \
                --batch-size "16" \
                --learning-rate "0.05" \
                --tau "0.6" \
                --pg-steps "5" \
                --nesterov \
                --epochs "100" \
                --wd "0.00005" \
                --warmup "0"

