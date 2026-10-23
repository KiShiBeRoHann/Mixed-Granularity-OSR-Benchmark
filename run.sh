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

# ---- 解析 EXTRA_ARGS 里的 --num_layers / --eval_view ----
# 兼容 "--num_layers 3" 与 "--num_layers=3" 两种写法（原实现只做子串匹配，
# 写成 = 形式时会被静默忽略 → 日志落进错误目录，与产物实际落盘位置不一致）
ARGS_ARR=("${@:3}")
NUM_LAYERS=""
EVAL_VIEW="V4"
SEEDS=(42 43 44 45 46)
FILTERED=()          # 剔除 --seeds 后，其余参数原样透传给训练脚本
i=0
while [ $i -lt ${#ARGS_ARR[@]} ]; do
    a="${ARGS_ARR[$i]}"
    case "$a" in
        --num_layers=*) NUM_LAYERS="${a#--num_layers=}"; FILTERED+=("$a") ;;
        --num_layers)   NUM_LAYERS="${ARGS_ARR[$((i + 1))]:-}"; FILTERED+=("$a" "${ARGS_ARR[$((i + 1))]:-}"); i=$((i + 1)) ;;
        --eval_view=*)  EVAL_VIEW="${a#--eval_view=}"; FILTERED+=("$a") ;;
        --eval_view)    EVAL_VIEW="${ARGS_ARR[$((i + 1))]:-V4}"; FILTERED+=("$a" "${ARGS_ARR[$((i + 1))]:-V4}"); i=$((i + 1)) ;;
        # --seeds 是 run.sh 自己的参数（训练脚本没有这个参数），必须从透传列表里剔除
        --seeds)        SEEDS=(); i=$((i + 1))
                        while [ $i -lt ${#ARGS_ARR[@]} ] && [[ "${ARGS_ARR[$i]}" != --* ]]; do
                            SEEDS+=("${ARGS_ARR[$i]}"); i=$((i + 1))
                        done
                        i=$((i - 1)) ;;
        *)              FILTERED+=("$a") ;;
    esac
    i=$((i + 1))
done
EXTRA_ARGS="${FILTERED[*]}"

LAYERS_DIR=""
if [[ "$DATASET" != "cifar100" ]]; then
    LAYERS_DIR="/L2"  # 默认树状数据集走 L2（与各脚本 argparse 的 num_layers 默认值 2 一致）
    if [[ "$NUM_LAYERS" == "3" ]]; then
        LAYERS_DIR="/L3"
    elif [[ "$NUM_LAYERS" == "2d" || "$NUM_LAYERS" == "2df" || "$NUM_LAYERS" == "3fc" ]]; then
        LAYERS_DIR="/L${NUM_LAYERS}"  # 从 L3 派生的两层划分（层数消融专用），与既有 L2/L3 完全隔离
    fi
fi

# 评估靶场视图隔离：非默认视图(V1/V2/V3)写入独立日志目录，避免覆盖 V4 的历史日志
# 与 datasets/__init__.py 的硬校验保持一致（视图只在明确支持的数据集+L3 下有效）：
#   aircraft + L3 : V1 / V2 / V3 / V4
#   imagenet + L3 : V1 / V2 / V3 / V4（V1/V2 的 gen 集只有 50~500 张，出表要看 n_gen_medium）
#   其它          : 仅默认 V4
# 不校验会生成"看似做过视图对照、数值其实与 V4 相同"的日志目录
VIEW_DIR=""
if [[ "$EVAL_VIEW" != "V4" ]]; then
    SUPPORTED="no"
    case "$EVAL_VIEW" in
        V1|V2|V3) ;;
        *) echo "❌ 未知 --eval_view $EVAL_VIEW（应取 V1/V2/V3/V4）"; exit 1 ;;
    esac
    if [[ "$NUM_LAYERS" == "3" ]]; then
        case "$DATASET" in
            aircraft|imagenet) SUPPORTED="yes" ;;
        esac
    fi
    if [[ "$SUPPORTED" != "yes" ]]; then
        echo "❌ --eval_view $EVAL_VIEW 在 dataset=$DATASET / num_layers=${NUM_LAYERS:-2} 下不支持"
        echo "   支持范围: aircraft+L3 / imagenet+L3 → V1~V4 | 其它 → 仅默认 V4"
        exit 1
    fi
    VIEW_DIR="/view_${EVAL_VIEW}"
fi

LOG_DIR="experiment_logs/${DATASET}${LAYERS_DIR}${VIEW_DIR}/${METHOD_NAME}"
mkdir -p "$LOG_DIR"

if [[ "$EXTRA_ARGS" == *"--include_unseen"* ]]; then
    LOG_FILE="${LOG_DIR}/include_unseen.log"
    MODE_STR="Include Unseen"
else
    LOG_FILE="${LOG_DIR}/standard.log"
    MODE_STR="Standard"
fi

# SEEDS 已在参数解析处初始化/被 --seeds 覆盖，此处不再重置

echo "==================================================" > $LOG_FILE
echo "方法: $SCRIPT_NAME" >> $LOG_FILE
echo "数据集: $DATASET${LAYERS_DIR}" >> $LOG_FILE
echo "运行模式: $MODE_STR" >> $LOG_FILE
echo "额外参数: $EXTRA_ARGS" >> $LOG_FILE
echo "==================================================" >> $LOG_FILE

# 5. 开始循环
for seed in "${SEEDS[@]}"; do
    echo -e "\n>>> 正在运行 SEED: $seed" >> $LOG_FILE
    
    python $SCRIPT_NAME --dataset $DATASET --seed $seed $EXTRA_ARGS >> $LOG_FILE 2>&1
    
    echo "✅ SEED $seed 运行完毕." >> $LOG_FILE
    sleep 2
done

echo -e "\n所有实验运行完毕($SCRIPT_NAME on $DATASET)" >> $LOG_FILE