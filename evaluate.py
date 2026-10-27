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
    data_path: str = "data/test.json",
    num_samples: int = 100,
    output_path: str = "eval_results.json",
    checkpoint_path: str = None,
    use_pretrained: bool = False,
):
    import argparse
    from load_weights import load_pretrained_weights

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # 1. Load exact test split
    if not os.path.exists(data_path):
        from create_splits import create_reproducible_splits
        print(f"{data_path} not found. Generating reproducible splits ...")
        create_reproducible_splits()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_data = data[:num_samples]
    print(f"Loaded {len(test_data)} test samples from {data_path}.")

    # 2. Load Model (either un-finetuned pretrained baseline or trained checkpoint)
    enc = tiktoken.get_encoding("gpt2")
    model = GPT2Model(GPTConfig())

    if use_pretrained:
        print("★ Evaluating UN-FINETUNED BASELINE (OpenAI GPT-2)...")
        model = load_pretrained_weights(model)
        if output_path == "eval_results.json":
            output_path = "eval_baseline_results.json"
    else:
        ckpt_path = checkpoint_path or "checkpoints/instruct_gpt2_best.pt"
        if not os.path.exists(ckpt_path):
            ckpt_path = "checkpoints/instruct_gpt2.pt"

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found at '{ckpt_path}'. Run training first, or run with '--use-pretrained' to evaluate base GPT-2."
            )

        print(f"★ Evaluating FINE-TUNED MODEL from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 3. Generate responses using greedy argmax
    results = []
    print(f"Generating responses for {len(test_data)} samples...")
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
        if i % 10 == 0 or i == len(test_data):
            print(f"  Processed [{i}/{len(test_data)}]")

    # 4. Save results to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Evaluation generation complete! Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate InstructGPT-2 on held-out test split")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (defaults to instruct_gpt2_best.pt)")
    parser.add_argument("--use-pretrained", action="store_true", help="Evaluate base un-finetuned GPT-2 as a comparison baseline")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of test samples to evaluate")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Path to output JSON")
    args = parser.parse_args()

    run_evaluation(
        num_samples=args.num_samples,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        use_pretrained=args.use_pretrained
    )
