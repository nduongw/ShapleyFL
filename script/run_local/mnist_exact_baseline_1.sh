TASK="mnist_classification"
DIST=1
SKEW=0.5
NUM_CLIENTS=12
SEED=0
python generate_fedtask.py --benchmark $TASK --dist $DIST --skew $SKEW --num_clients $NUM_CLIENTS --seed $SEED

TASK="${TASK}_cnum${NUM_CLIENTS}_dist${DIST}_skew${SKEW}_seed${SEED}"
GPU_IDS=( 0 )
NUM_THREADS=1
BATCH_SIZE=64
NUM_ROUNDS=50
PROPORTION=1.0
    
python main.py \
    --task $TASK \
    --model cnn \
    --algorithm sv_fedavg \
    --num_rounds $NUM_ROUNDS \
    --num_epochs 2 \
    --learning_rate 0.01 \
    --lr_scheduler 0 \
    --learning_rate_decay 1.0 \
    --proportion 1 \
    --batch_size $BATCH_SIZE \
    --eval_interval 1 \
    --gpu $GPU_IDS \
    --num_threads $NUM_THREADS \
    --aggregate weighted_scale \
    --sample full \
    --exact \
    --round_calSV 1 \
