#!/usr/bin/env bash

LOG=logs/a2d_sent_new/deeplab_cmpc_video_mm_tgraph_allvec
mkdir -p ${LOG}
now=$(date +"%Y%m%d_%H%M%S")

python -u trainval_video.py \
-m train \
-d a2d_sent_new \
-t train \
-n CMPC_video_mm_tgraph_allvec \
-i 400000 \
-s 20000 \
-st 380000 \
-lrd 400000 \
-emb \
-g 2 \
-f ckpts/a2d_sent_new/deeplab_cmpc_video_mm_tgraph_allvec 2>&1 | tee ${LOG}/train_$now.txt

python -u trainval_video.py \
-m test \
-d a2d_sent_new \
-t test \
-n CMPC_video_mm_tgraph_allvec \
-i 360000 \
-c \
-emb \
-g 2 \
-f ckpts/a2d_sent_new/deeplab_cmpc_video_mm_tgraph_allvec 2>&1 | tee ${LOG}/test_$now.txt
