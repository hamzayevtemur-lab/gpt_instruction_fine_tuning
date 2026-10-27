import torch
from transformers import GPT2LMHeadModel as HF_GPT2

@torch.no_grad()
def load_pretrained_weights(model, hf_name="openai-community/gpt2"):

    print(f"Loading weights from {hf_name} ...")
    hf_model = HF_GPT2.from_pretrained(hf_name)

    hf_sd    = hf_model.state_dict()
    our_sd   = model.state_dict()

    transposed = ("c_attn.weight", "c_proj.weight", "c_fc.weight")

    loaded, skipped = 0, 0
    for key, val in hf_sd.items():
        if key not in our_sd:
            skipped += 1
            continue

        if any(key.endswith(t) for t in transposed):
            our_sd[key].copy_(val.t())
        else:
            our_sd[key].copy_(val)

        loaded += 1

    model.load_state_dict(our_sd)
    print(f"Loaded {loaded} | Skipped {skipped}")
    return model
