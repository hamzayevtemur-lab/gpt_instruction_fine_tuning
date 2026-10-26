import os
import json
import torch
import tiktoken
from config import CHECKPOINT_PATH, MAX_LENGTH
from formatter import format_prompt, format_response_header
from gpt import GPT2Model, GPTConfig


def clean_response(raw_text: str) -> str:
    """Extracts only the generated response after '### Response:' and cuts off at <|endoftext|>."""
    if "### Response:\n" in raw_text:
        raw_text = raw_text.split("### Response:\n", 1)[1]
    return raw_text.replace("<|endoftext|>", "").strip()


@torch.no_grad()
def generate_response(model, enc, prompt: str, max_new_tokens: int = 128, device: str = "cpu") -> str:
    input_ids = enc.encode(prompt)
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
    eos_id = enc.eot_token

    for _ in range(max_new_tokens):
        context = tokens[:, -MAX_LENGTH:] if tokens.size(1) > MAX_LENGTH else tokens
        logits = model(context)[:, -1, :]
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)

        if next_tok.item() == eos_id:
            break

        tokens = torch.cat([tokens, next_tok], dim=1)

    generated_ids = tokens[0, len(input_ids):].tolist()
    if eos_id in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(eos_id)]

    return enc.decode(generated_ids).strip()


def run_evaluation(data_path: str = "data/alpaca_data.json", num_samples: int = 100, output_path: str = "eval_results.json"):
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Using device: {device}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_data = data[-num_samples:]
    print(f"Loaded {len(test_data)} test samples.")

    enc = tiktoken.get_encoding("gpt2")
    model = GPT2Model(GPTConfig())

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

    print(f"Loading checkpoint from {CHECKPOINT_PATH} ...")
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    results = []
    print(f"Generating responses for {num_samples} samples...")

    for i, entry in enumerate(test_data, 1):
        instruction = entry.get("instruction", "")
        input_text = entry.get("input", "")
        reference = entry.get("output", "")

        prompt = format_response_header(instruction, input_text)
        model_response = generate_response(model, enc, prompt, max_new_tokens=128, device=device)

        results.append({
            "id": i,
            "instruction": instruction,
            "input": input_text,
            "reference_output": reference,
            "model_response": model_response
        })

        if i % 10 == 0 or i == num_samples:
            print(f"  Processed [{i}/{num_samples}]")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Evaluation complete! Saved to {output_path}")


if __name__ == "__main__":
    run_evaluation()
