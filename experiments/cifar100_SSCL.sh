WORKERS = 8
EPOCH = 10
THRESHOLD = 0.5
MEMORY = 400
BATCH_SIZE = 16
UL_BATCH_SIZE = 32
LABELED_SAMPLES = 500

python -m main.py --workers $WORKERS --epoch $EPOCH --threshold $THRESHOLD --memory $MEMORY --batch_size $BATCH_SIZE --ul_batch_size $UL_BATCH_SIZE --labeled_samples $LABELED_SAMPLES