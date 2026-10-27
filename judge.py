import json
import os
import re
import time
import google.generativeai as genai


# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_key = input("Enter your Gemini API key: ").strip()


genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-3.8-flash")


JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing the performance of a fine-tuned language model.
Evaluate the following model response against the instruction and ground truth reference.

[Instruction]:
{instruction}

[Input]:
{input}

[Reference (Ground Truth)]:
{reference}

[Model Response]:
{response}

Evaluate the Model Response on a scale from 0 to 100 based on:
1. Instruction Following: Did it follow all instructions and constraints?
2. Correctness & Quality: Is the content accurate, coherent, and helpful?
3. Repetition & Hallucination: Did it repeat itself or produce garbage text?

Respond with ONLY a single integer number between 0 and 100. Do not write any other words or punctuation.
"""



def parse_score(text: str) -> int:
    """Extracts just the number from Gemini's response."""
    # Find any sequence of digits
    match = re.search(r"\b(100|[1-9]?[0-9])\b", text.strip())
    if match:
        return int(match.group(1))
    return 0



def run_judge(
    input_file: str = "eval_results.json",
    output_file: str = "judged_results.json",
):
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Results file '{input_file}' not found. Run evaluate.py first!"
        )

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} generated samples from {input_file}")
    print("Judging with Gemini 1.5 Flash...\n")


    judged_records = []
    scores = []

    for item in data:
        item_id = item["id"]
        instruction = item["instruction"]
        inp = item.get("input", "")
        reference = item["reference_output"]
        model_response = item["model_response"]

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            instruction=instruction,
            input=inp if inp else "None",
            reference=reference,
            response=model_response if model_response else "(Empty response)",
        )

        try:
            response = model.generate_content(prompt)
            judge_text = response.text.strip()
            score = parse_score(judge_text)
            scores.append(score)

            judged_records.append(
                {
                    "id": item_id,
                    "instruction": instruction,
                    "input": inp,
                    "model_response": model_response,
                    "reference": reference,
                    "judge_evaluation": judge_text,
                    "score": score,
                }
            )

            print(f"[{item_id:03d}/{len(data)}] Score: {score:3d}/100", flush=True)
            time.sleep(0.5)  # Stay within free tier rate limit

        except Exception as e:
            print(f"[{item_id:03d}/{len(data)}] Error: {e}", flush=True)
            time.sleep(2)


    # Save detailed grades
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(judged_records, f, indent=2, ensure_ascii=False)

    if scores:
        avg_score = sum(scores) / len(scores)
        print("\n" + "=" * 45)
        print(f" Final Average Score: {avg_score:.2f} / 100")
        print(f" Detailed report saved to: {output_file}")
        print("=" * 45)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model responses with Gemini LLM-as-a-judge")
    parser.add_argument("--input", type=str, default="checkpoints/eval_results.json", help="Path to input evaluation json")
    parser.add_argument("--output", type=str, default="results/judged_results.json", help="Path to output judged json")
    args = parser.parse_args()

    run_judge(input_file=args.input, output_file=args.output)





