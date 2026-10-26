import json
import os
import tiktoken
import torch
from config import CHECKPOINT_PATH, MAX_LENGTH
from formatter import format_response_header
from gpt.config import GPTConfig
from gpt.model import GPT2Model

def clean_response(raw_text: str)->str:
    """Extracts only the generated response after '### Response:' and cuts off at <|endoftext|>."""
    if "### Response:\n" in raw_text:
        raw_text = raw_text.split("### Response:\n", 1)[1]
    
    # Remove end of text marker if present
    raw_text = raw_text.replace("<|endoftext|>", "").strip()
    return raw_text



@torch.no_grad()
def generate_response(model, enc, prompt: str, max_new_tokens: int=128, device: str="cpu")->str:

    input_ids=enc.encode(prompt)
    input_tensor=torch.tensor([input_ids], dtype=torch.long, device=device)

    prompt_len = len(input_ids)
    for _ in range(max_new_tokens):
        # Crop to max context window if needed
        idx_cond = (
            input_tensor[:, -1024:]
            if input_tensor.size(1) > 1024
            else input_tensor
        )

        logits = model(idx_cond)
        next_token_logits = logits[:, -1, :]

        # Greedy decoding (argmax) for consistent evaluation
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if next_token.item() == 50256:  # <|endoftext|>
            break

        input_tensor = torch.cat((input_tensor, next_token), dim=1)

    generated_ids = input_tensor[0, prompt_len:].tolist()
    if 50256 in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(50256)]

    return enc.decode(generated_ids).strip()



def run_evaluation(
    data_path: str = "data/alpaca_data.json",
    num_samples: int = 100,
    output_path: str = "eval_results.json",
):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # 1. Load data
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use the last N samples as test set (held out from training)
    test_data = data[-num_samples:]
    print(f"Loaded {len(test_data)} test samples.")

    # 2. Load Model
    enc = tiktoken.get_encoding("gpt2")
    model = GPT2Model(GPTConfig())

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")
    print(f"Loading checkpoint from {CHECKPOINT_PATH} ...")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 3. Generate responses
    results = []

    print(f"Generating responses for {num_samples} samples...")
    for i, entry in enumerate(test_data, 1):
        instruction = entry.get("instruction", "")
        input_text = entry.get("input", "")
        reference = entry.get("output", "")
        prompt = format_response_header(instruction, input_text)
        model_response = generate_response(
            model, enc, prompt, max_new_tokens=128, device=device
        )
        results.append(
            {
                "id": i,
                "instruction": instruction,
                "input": input_text,
                "reference_output": reference,
                "model_response": model_response,
            }
        )
        if i % 10 == 0 or i == num_samples:
            print(f"  Processed [{i}/{num_samples}]")

    # 4. Save results to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Evaluation generation complete! Saved to {output_path}")
    
if __name__ == "__main__":
    run_evaluation()
