#!/usr/bin/bash
DEVICE=2
WORKERS=8
MODEL_NAME='Reduced_ResNet18'
EPOCH=10
THRESHOLD=0.5
MEMORY=400
BATCH_SIZE=128
UL_BATCH_SIZE=256
# #image per task = 2500 / labeled_samples=500(noisy 80%), labeled_samples=1000(noisy 60%)
LABELED_SAMPLES=500
# LABELED_SAMPLES=1000


python main.py --device $DEVICE --workers $WORKERS --model_name $MODEL_NAME --epoch $EPOCH --threshold $THRESHOLD --memory $MEMORY --batch_size $BATCH_SIZE --ul_batch_size $UL_BATCH_SIZE --labeled_samples $LABELED_SAMPLES