python main.py \
    --task synthetic_cnum16_dist12_skew1.0_seed1 \
    --model lr \
    --algorithm mp_fedavg \
    --num_rounds 200 \
    --num_epochs 5 \
    --learning_rate 0.01 \
    --lr_scheduler 0 \
    --learning_rate_decay 0.998 \
    --proportion 1 \
    --batch_size 10 \
    --eval_interval 1 \
    --gpu 0 \
    --num_threads_per_gpu 1

# python main.py \
#     --task cifar10_cnum16_dist2_skew1.0_seed0 \
#     --model resnet18 \
#     --algorithm mp_fedavg \
#     --num_rounds 200 \
#     --num_epochs 5 \
#     --learning_rate 0.01 \
#     --lr_scheduler 0 \
#     --learning_rate_decay 0.998 \
#     --proportion 1 \
#     --batch_size 10 \
#     --eval_interval 1 \
#     --gpu 1 \
#     --num_threads_per_gpu 1