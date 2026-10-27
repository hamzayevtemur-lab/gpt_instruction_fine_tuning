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



def calc_val_loss(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, (input_ids, labels) in enumerate(val_loader):
            if max_batches and i >= max_batches:
                break

            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Token-weighted loss calculation
            valid_mask = (shift_labels != -100)
            num_valid = valid_mask.sum().item()
            if num_valid == 0:
                continue

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum"
            )

            total_loss += loss.item()
            total_tokens += num_valid

    model.train()
    return (total_loss / total_tokens) if total_tokens > 0 else 0.0


def train():
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Train InstructGPT-2")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    args = parser.parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    cfg   = GPTConfig()
    model = GPT2Model(cfg)

    trainable_params = setup_for_finetuning(model, mode=FINETUNE_MODE,
                                             last_n_blocks=TRAIN_LAST_N_BLOCKS,
                                             freeze_embeddings=FREEZE_EMBEDDINGS)

    optimizer = torch.optim.AdamW(
        trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(checkpoint_dir, "instruct_gpt2_latest.pt")
    best_path   = os.path.join(checkpoint_dir, "instruct_gpt2_best.pt")

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")

    if args.resume and os.path.exists(latest_path):
        print(f"Resuming training from: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resumed at Epoch {start_epoch}, Step {global_step} (Best Val Loss: {best_val_loss:.4f})")
    else:
        model = load_pretrained_weights(model)

    model.to(device)

    start_time = time.time()
    print("\n── Training started ──")
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
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

            if torch.isnan(loss):
                print(f"Warning: Non-finite loss encountered at Step {step} — skipping step.")
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            global_step += 1
            if step % 50 == 0:
                print(f"Epoch {epoch}/{NUM_EPOCHS} | Step {step:05d} | Loss: {loss.item():.4f}")

            if global_step % SAVE_EVERY_N_STEPS == 0:
                save_dict = {
                    "epoch": epoch,
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                }
                torch.save(save_dict, latest_path)
                print(f"  ✓ Latest recovery checkpoint saved at step {global_step}")

        val_loss = calc_val_loss(model, val_loader, device)
        print(f"\n── Epoch {epoch} done | Val Loss: {val_loss:.4f}")

        # Check if this is the best model so far
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
            }, best_path)
            print(f"  ★ New best model saved! Val Loss: {best_val_loss:.4f} -> {best_path}")

        # Also update latest recovery checkpoint
        torch.save({
            "epoch": epoch + 1,
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }, latest_path)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes.")
    print(f"Best Checkpoint: {best_path} (Val Loss: {best_val_loss:.4f})")
    print(f"Latest Checkpoint: {latest_path}")


if __name__ == "__main__":
    train()
    