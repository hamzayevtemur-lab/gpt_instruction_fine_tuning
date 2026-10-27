import tiktoken
import torch

from formatter import format_prompt, format_response_header
from dataset import AlpacaDataset, custom_collate_fn


def test_token_boundary_alignment():
    """
    Test 1: Verify whether tokenizing header separately matches tokenizing full text.
    BPE tokenizers can merge characters across string concatenation boundaries.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    
    instruction = "Write a haiku about code."
    inp = ""
    response = "Clean code reads so well\nSilent bugs are fixed today\nTests pass with delight"
    
    full_text = format_prompt(instruction, inp, response)
    header_text = format_response_header(instruction, inp)
    
    full_tokens = tokenizer.encode(full_text)
    header_tokens = tokenizer.encode(header_text)
    
    header_len = len(header_tokens)
    
    actual_prefix = full_tokens[:header_len]
    
    print(f"\nHeader tokens: {header_tokens}")
    print(f"Full prefix:   {actual_prefix}")
    
    assert actual_prefix == header_tokens, (
        f"Boundary mismatch! Header tokens: {header_tokens} vs Full prefix: {actual_prefix}"
    )


def test_response_masking_values():
    """
    Test 2: Verify that labels before response start are strictly -100,
    and labels inside response match the actual input_ids.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    sample_data = [{
        "instruction": "Identify the primary color.",
        "input": "Grass",
        "output": "Green."
    }]
    
    dataset = AlpacaDataset(sample_data, tokenizer, max_length=128)
    input_ids, labels = dataset[0]
    
    header_text = format_response_header("Identify the primary color.", "Grass")
    header_len = len(tokenizer.encode(header_text))
    
    # All tokens up to header_len MUST be -100
    for i in range(header_len):
        assert labels[i] == -100, f"Token at {i} should be -100, got {labels[i]}"
        
    # The tokens after header_len MUST match input_ids
    for i in range(header_len, len(input_ids)):
        assert labels[i] == input_ids[i], f"Mismatch at {i}: input {input_ids[i]} vs label {labels[i]}"
        
    # Verify that the response text can be accurately decoded from non-masked labels
    response_tokens = [tok for tok in labels[header_len:] if tok != -100]
    decoded_response = tokenizer.decode(response_tokens).strip()
    assert "Green." in decoded_response


def test_eos_token_present_in_labels():
    """
    Test 3: Ensure <|endoftext|> (50256) is included in the target labels,
    so the model learns when to STOP generating.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    sample_data = [{
        "instruction": "Say hello.",
        "input": "",
        "output": "Hello!"
    }]
    
    dataset = AlpacaDataset(sample_data, tokenizer, max_length=128)
    input_ids, labels = dataset[0]
    
    eos_id = tokenizer.eot_token
    assert eos_id in labels, "Crucial failure: <|endoftext|> is missing from target labels!"


def test_truncation_filters_empty_responses():
    """
    Test 4: Verify that when an instruction is too long and max_length cuts off
    the entire response, the dataset SAFELY discards the sample instead of
    feeding a batch with 100% masked (-100) labels.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    
    long_instruction = "Explain quantum gravity with extensive math and derivations: " * 5
    sample_data = [{
        "instruction": long_instruction,
        "input": "",
        "output": "It is complex."
    }]
    
    # Very short max_length cuts off before the response starts
    dataset_short = AlpacaDataset(sample_data, tokenizer, max_length=20)
    assert len(dataset_short) == 0, "Dataset should have skipped sample with no valid response tokens!"
    
    # Sufficient max_length retains the sample with valid response tokens
    dataset_long = AlpacaDataset(sample_data, tokenizer, max_length=150)
    assert len(dataset_long) == 1, "Dataset should retain sample when response fits!"
    _, labels = dataset_long[0]
    valid_targets = [l for l in labels if l != -100]
    assert len(valid_targets) > 0, "Retained sample must have valid response targets!"


def test_dynamic_collate_fn():
    """
    Test 5: Verify that custom_collate_fn dynamically pads a batch correctly.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = [
        ([1, 2, 3], [-100, -100, 3]),
        ([1, 2, 3, 4, 5], [-100, -100, 3, 4, 5]),
        ([1, 2], [-100, 2])
    ]
    
    x, y = custom_collate_fn(batch, pad_token_id=tokenizer.eot_token)
    
    assert x.shape == (3, 5)
    assert y.shape == (3, 5)
    
    assert x[0, 3].item() == tokenizer.eot_token
    assert x[0, 4].item() == tokenizer.eot_token
    assert y[0, 3].item() == -100
    assert y[0, 4].item() == -100
