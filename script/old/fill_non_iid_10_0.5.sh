FEDTASK='synthetic_classification_cnum10_dist10_skew0.5_seed0'
for N in 6 7 8 9 10
do
    echo $N
    python fill.py \
        --fedtask $FEDTASK \
        --exact_dir "exact-${N}"
done