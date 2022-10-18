#!/usr/bin/bash
WORKERS=8
# MODEL_NAME='ResNet18'Reduced_ResNet18
MODEL_NAME='Reduced_ResNet18'
EPOCH=10
THRESHOLD=0.5
MEMORY=400
BATCH_SIZE=16
UL_BATCH_SIZE=32
LABELED_SAMPLES=500


python main.py --workers $WORKERS --model_name $MODEL_NAME --epoch $EPOCH --threshold $THRESHOLD --memory $MEMORY --batch_size $BATCH_SIZE --ul_batch_size $UL_BATCH_SIZE --labeled_samples $LABELED_SAMPLES