import os
import torch
import tiktoken
from gpt import GPT2Model, GPTConfig
from formatter import format_response_header
from config import CHECKPOINT_PATH, MAX_LENGTH


@torch.no_grad()
def generate(model, tokenizer, instruction, inp="",
             max_new_tokens=200, temperature=0.8, top_k=50, device="cpu"):

    model.eval()

    header=format_response_header(instruction, inp)

    tokens=tokenizer.encode(header)

    tokens=torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    eos_id = tokenizer.eot_token  # 50256

    prompt_len = tokens.size(1)
    for _ in range(max_new_tokens):
        context = tokens[:, -MAX_LENGTH:]

        logits  = model(context)[:, -1, :]
        logits  = logits / temperature

        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            logits = logits.masked_fill(logits < values[:, -1:], float("-inf"))

        probs    = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        tokens   = torch.cat([tokens, next_tok], dim=1)

        if next_tok.item() == eos_id:
            break

    out = tokens[0, prompt_len:].tolist()
    if eos_id in out:
        out = out[:out.index(eos_id)]

    return tokenizer.decode(out).strip()




def main():
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )

    cfg   = GPTConfig()
    model = GPT2Model(cfg)

    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"Loaded: {CHECKPOINT_PATH}\n")
    else:
        print("No checkpoint found — using random weights.\n")

    model.to(device)
    tokenizer = tiktoken.get_encoding("gpt2")

    print("=== InstructGPT2 — Interactive ===")
    print("Type 'quit' to exit.\n")

    while True:
        instruction = input("Instruction: ").strip()
        if instruction.lower() == "quit":
            break
        inp = input("Input (Enter to skip): ").strip()
        response = generate(model, tokenizer, instruction, inp, device=device)
        print(f"\nResponse:\n{response}\n")
        print("-" * 50)
        
if __name__ == "__main__":
    main()



    