#!/bin/bash

#training params
EPOCHS=50
BATCHSIZE=64

METHODS=('baseline' 'ewc' 'buffer')
# METHODS=('baseline_inc_tasks' 'baseline_alltasks')


ROOT_DIR=$4
MODEL=$3
DATASET_TYPE=$2 #('disjoint' or 'strict')
SEED=$1

TRAIN_PATH_TASK0=$ROOT_DIR'/'$DATASET_TYPE'/train/images/t0'
TRAIN_PATH_TASK1=$ROOT_DIR'/'$DATASET_TYPE'/train/images/t1'
TRAIN_PATH_TASK2=$ROOT_DIR'/'$DATASET_TYPE'/train/images/t2'

VAL_PATH_TASK0=$ROOT_DIR'/'$DATASET_TYPE'/val/images/t0'
VAL_PATH_TASK1=$ROOT_DIR'/'$DATASET_TYPE'/val/images/t1'
VAL_PATH_TASK2=$ROOT_DIR'/'$DATASET_TYPE'/val/images/t2'

TEST_PATH_TASK0=$ROOT_DIR'/'$DATASET_TYPE'/test/images/t0'
TEST_PATH_TASK1=$ROOT_DIR'/'$DATASET_TYPE'/test/images/t1'
TEST_PATH_TASK2=$ROOT_DIR'/'$DATASET_TYPE'/test/images/t2'

TEST_PATH_GLOBAL=$ROOT_DIR'/global/test/images/t0'
TRAIN_PATH_GLOBAL=$ROOT_DIR'/global/train/images/t0'
VAL_PATH_GLOBAL=$ROOT_DIR'/global/val/images/t0'

for method in "${METHODS[@]}"
do

    if [[ "$method" == "baseline_inc_tasks" || "$method" == "baseline_alltasks" ]]; then

        python train_files/train_all_tasks.py --epochs $EPOCHS --batch_size $BATCHSIZE --model_name $MODEL --dataset_type $DATASET_TYPE --results_dir $ROOT_DIR \
        --train_path_task0 $TRAIN_PATH_TASK0 --train_path_task1 $TRAIN_PATH_TASK1 --train_path_task2 $TRAIN_PATH_TASK2 \
        --val_path_task0 $VAL_PATH_TASK0 --val_path_task1 $VAL_PATH_TASK1 --val_path_task2 $VAL_PATH_TASK2 \
        --test_path_task0 $TEST_PATH_TASK0 --test_path_task1 $TEST_PATH_TASK1 --test_path_task2 $TEST_PATH_TASK2 \
        --test_path_global $TEST_PATH_GLOBAL --method $method --seed $SEED

    else 
        python train_files/train.py --epochs $EPOCHS --batch_size $BATCHSIZE --model_name $MODEL --dataset_type $DATASET_TYPE \
        --train_path_task0 $TRAIN_PATH_TASK0 --train_path_task1 $TRAIN_PATH_TASK1 --train_path_task2 $TRAIN_PATH_TASK2 \
        --val_path_task0 $VAL_PATH_TASK0 --val_path_task1 $VAL_PATH_TASK1 --val_path_task2 $VAL_PATH_TASK2 \
        --test_path_task0 $TEST_PATH_TASK0 --test_path_task1 $TEST_PATH_TASK1 --test_path_task2 $TEST_PATH_TASK2 \
        --test_path_global $TEST_PATH_GLOBAL --method $method --seed $SEED --results_dir $ROOT_DIR
    fi
    
done

