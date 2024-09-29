#!/bin/zsh
NOW=`date +'%I_%M_%d_%m'`

EMBEDDING_MODEL=thenlper_gte-large
DATASET_PARTIAL_PATH="../data_preprocessing/PetSupplies/Pet"
DATASET="${DATASET_PARTIAL_PATH}.txt"
ITEM_FREQ="${DATASET_PARTIAL_PATH}-train_item_freq.txt"
SIMILARITY_INDICES="${DATASET_PARTIAL_PATH}-similarity-indices-${EMBEDDING_MODEL}.pt"
SIMILARITY_VALUES="${DATASET_PARTIAL_PATH}-similarity-values-${EMBEDDING_MODEL}.pt"


SIMILARITY_THREHOLD=0.7
TEMPERATURE=0.5
LAMBDA=0.6
LAMBDA_SCHEDULING=LINEAR
LAMBDA_WARMPUP=1000
LAMBDA_STEPS=70000
MAX_LEN=50
BATCH_SIZE=128
LR=0.0001
DROPOUT=0.5
NUM_BLOCKS=2
EPOCHS=200
DEVICE="cuda:0"
HIDDEN_DIM=50
TRAIN_DIR="results/pet/${NOW}"

python main.py --dataset ${DATASET}\
               --item_frequency ${ITEM_FREQ}\
               --similarity_indices ${SIMILARITY_INDICES}\
               --similarity_values ${SIMILARITY_VALUES}\
               --similarity_threshold ${SIMILARITY_THREHOLD}\
               --temperature ${TEMPERATURE}\
               --lambd ${LAMBDA}\
               --lambd_scheduling "${LAMBDA_SCHEDULING}"\
               --lambd_warmup_steps ${LAMBDA_WARMPUP}\
               --lambd_steps ${LAMBDA_STEPS}\
               --batch_size ${BATCH_SIZE}\
               --lr ${LR}\
               --maxlen ${MAX_LEN}\
               --dropout_rate ${DROPOUT}\
               --num_blocks ${NUM_BLOCKS}\
               --num_epochs ${EPOCHS}\
               --hidden_units ${HIDDEN_DIM}\
               --train_dir ${TRAIN_DIR}\
               --device ${DEVICE}