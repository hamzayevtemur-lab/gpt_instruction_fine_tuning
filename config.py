BATCH_SIZE          = 4
MAX_LENGTH          = 256     # shorter = faster forward pass
LEARNING_RATE       = 3e-5
NUM_EPOCHS          = 3
WEIGHT_DECAY        = 0.01
GRAD_CLIP           = 1.0
SAVE_EVERY_N_STEPS  = 500
CHECKPOINT_PATH     = "checkpoints/instruct_gpt2.pt"
VAL_RATIO           = 0.05
NUM_SAMPLES         = 10000   # use subset for fast training

# Fine-tuning strategy:
#   'full'    -> train all parameters
#   'partial' -> freeze bottom layers, train only last N blocks + ln_f
FINETUNE_MODE       = "partial"
TRAIN_LAST_N_BLOCKS = 2       # 2 blocks ≈ 14M params (fast)

# If True: freeze lm_head + wte (embedding table = 38.6M params)
# Recommended True for speed — model already knows how to produce words
FREEZE_EMBEDDINGS   = True
