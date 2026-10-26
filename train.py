import os
import torch
import torch.nn.functional as F
from gpt import GPT2Model, GPTConfig
from load_weights import load_pretrained_weights
from dataset import get_dataloaders
from config import (
    BATCH_SIZE, MAX_LENGTH, LEARNING_RATE,
    NUM_EPOCHS, WEIGHT_DECAY, GRAD_CLIP,
    SAVE_EVERY_N_STEPS, CHECKPOINT_PATH, VAL_RATIO,
    FINETUNE_MODE, TRAIN_LAST_N_BLOCKS, FREEZE_EMBEDDINGS,
    NUM_SAMPLES
)


def setup_for_finetuning(model, mode="full", last_n_blocks=2, freeze_embeddings=True):
    """
    mode options:
      - 'full'    : train all parameters (best quality, slower)
      - 'partial' : freeze bottom layers, only train last N blocks + ln_f
    freeze_embeddings:
      - True  : freeze lm_head + wte (saves 38.6M params from being trained)
      - False : also train the output head and token embeddings
    """
    if mode == "full":
        trainable = list(model.parameters())
        print(f"[full] All {sum(p.numel() for p in trainable):,} parameters trainable.")
        return trainable

    elif mode == "partial":
        # Step 1: Freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Step 2: Unfreeze last N transformer blocks
        for block in model.transformer["h"][-last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # Step 3: Unfreeze final LayerNorm
        for param in model.transformer["ln_f"].parameters():
            param.requires_grad = True

        # Step 4: Optionally unfreeze lm_head + wte
        # NOTE: lm_head and wte share the same weight tensor (weight tying)
        # Unfreezing one also unfreezes the other — they add 38.6M params!
        if not freeze_embeddings:
            for param in model.lm_head.parameters():
                param.requires_grad = True

        trainable = [p for p in model.parameters() if p.requires_grad]
        total     = sum(p.numel() for p in model.parameters())
        t_count   = sum(p.numel() for p in trainable)
        emb_status = "frozen" if freeze_embeddings else "trainable"
        print(f"[partial] Trainable: {t_count:,} / {total:,} params "
              f"({100*t_count/total:.1f}%) — "
              f"last {last_n_blocks} blocks + ln_f | embeddings: {emb_status}")
        return trainable

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'full' or 'partial'.")



def calc_val_loss(model, val_loader, device, max_batches=20):
    model.eval()
    total, count=0.0, 0
    with torch.no_grad():
        for i, (input_ids, labels) in enumerate(val_loader):
            if i >= max_batches:
                break

            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            total += loss.item()
            count += 1

    model.train()

    return total/count if count>0 else 0.0



def train():
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        val_ratio=VAL_RATIO,
        num_samples=NUM_SAMPLES
    )

    cfg   = GPTConfig()
    model = GPT2Model(cfg)
    model = load_pretrained_weights(model)
    model.to(device)

    # Partial or full fine-tuning
    trainable_params = setup_for_finetuning(model, mode=FINETUNE_MODE,
                                             last_n_blocks=TRAIN_LAST_N_BLOCKS,
                                             freeze_embeddings=FREEZE_EMBEDDINGS)

    optimizer = torch.optim.AdamW(
        trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    global_step = 0

    print("\n── Training started ──")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        for step, (input_ids, labels) in enumerate(train_loader, 1):
            input_ids, labels = input_ids.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = model(input_ids)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            global_step += 1
            if step % 50 == 0:
                print(f"Epoch {epoch}/{NUM_EPOCHS} | Step {step:05d} | Loss: {loss.item():.4f}")

            if global_step % SAVE_EVERY_N_STEPS == 0:
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"  ✓ Checkpoint saved at step {global_step}")

        val_loss = calc_val_loss(model, val_loader, device)
        print(f"\n── Epoch {epoch} done | Val Loss: {val_loss:.4f}\n")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Done! Saved → {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
    