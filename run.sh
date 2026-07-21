#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "❌ 参数错误！"
    echo "💡 正确用法: $0 <训练脚本.py> <数据集> [额外参数...]"
    echo "💡 示例操作: $0 train_a2pt.py aircraft --num_layers 2 --include_unseen"
    exit 1
fi

SCRIPT_NAME=$1
DATASET=$2
EXTRA_ARGS="${@:3}" 

METHOD_NAME="${SCRIPT_NAME%.*}"         
mkdir -p experiment_logs                

LAYERS_DIR=""
if [[ "$DATASET" != "cifar100" ]]; then
    LAYERS_DIR="/L2"  # 默认树状数据集走 L2
    if [[ "$EXTRA_ARGS" == *"--num_layers 3"* ]]; then
        LAYERS_DIR="/L3"
    fi
fi

LOG_DIR="experiment_logs/${DATASET}${LAYERS_DIR}/${METHOD_NAME}"
mkdir -p "$LOG_DIR"

if [[ "$EXTRA_ARGS" == *"--include_unseen"* ]]; then
    LOG_FILE="${LOG_DIR}/include_unseen.log"
    MODE_STR="Include Unseen"
else
    LOG_FILE="${LOG_DIR}/standard.log"
    MODE_STR="Standard"
fi

if [ "$DATASET" == "cifar100" ]; then
    SEEDS=(42 43 44 45)
else
    SEEDS=(42 43 44 45 46)
fi

echo "==================================================" > $LOG_FILE
echo "方法: $SCRIPT_NAME" >> $LOG_FILE
echo "数据集: $DATASET${LAYERS_DIR}" >> $LOG_FILE
echo "运行模式: $MODE_STR" >> $LOG_FILE
echo "==================================================" >> $LOG_FILE

# 5. 开始循环
for seed in "${SEEDS[@]}"; do
    echo -e "\n>>> 正在运行 SEED: $seed" >> $LOG_FILE
    
    python $SCRIPT_NAME --dataset $DATASET --seed $seed $EXTRA_ARGS >> $LOG_FILE 2>&1
    
    echo "✅ SEED $seed 运行完毕." >> $LOG_FILE
    sleep 2
done

echo -e "\n所有实验运行完毕($SCRIPT_NAME on $DATASET)" >> $LOG_FILE