#!/usr/bin/env bash

LOG=logs/unc/cmpc_model
mkdir -p ${LOG}
now=$(date +"%Y%m%d_%H%M%S")

python -u trainval_model.py \
-m train \
-d unc \
-t train \
-n CMPC_model \
-emb \
-f ckpts/unc/cmpc_model 2>&1 | tee ${LOG}/train_$now.txt

python -u trainval_model.py \
-m test \
-d unc \
-t val \
-n CMPC_model \
-i 700000 \
-c \
-emb \
-f ckpts/unc/cmpc_model 2>&1 | tee ${LOG}/test_val_$now.txt
