from transformers import (
    AutoTokenizer, AutoModelForCausalLM
)
import torch
import numpy as np
import re

from typing import List, Dict, Tuple, Optional

def load_model(model_name):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left"  # Important for batch generation
    )
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="../deception-detection/data/huggingface/",
    )
    model.eval()

    return model, tokenizer
    

def format_prompt_for_model(prompt: str, model_name: str) -> str:
#     """Format prompt according to model's expected format."""
    # Model-specific formatting
    if "llama" in model_name.lower():
        # Llama 3.1 format
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
        # Mistral format
        return f"[INST] {prompt} [/INST]"
    
    elif "qwen" in model_name.lower():
        # Qwen format - they often work well with simple prompts
        return prompt
    else:
        assert False, f"unsupported model: {model_name}"

def extract_answer(text: str) -> int:
    """Extract the numerical answer from model output."""
    # First, try to find number in parentheses
    match = re.search(r'\((\d+)\)', text)
    if match:
        return int(match.group(1))
    
    # If not found, look for patterns like "Answer: 3" or just "3"
    # But only in the first line or two to avoid random numbers
    lines = text.strip().split('\n')[:2]
    for line in lines:
        # Look for "answer is X" patterns
        match = re.search(r'answer\s*(?:is|:)?\s*(\d+)', line.lower())
        if match:
            return int(match.group(1))
        
        # Look for standalone number at the end
        match = re.search(r'(\d+)\s*$', line)
        if match:
            return int(match.group(1))
    
    return -1 


def analyze_errors(predictions: List[Dict]) -> Dict:
    """Analyze error patterns in predictions."""
    errors = [p for p in predictions if p['predicted'] != p['true']]
    
    if not errors:
        return {}
    
    # Calculate error statistics
    true_counts = [e['true'] for e in errors]
    predicted_counts = [e['predicted'] for e in errors if e['predicted'] != -1]
    
    analysis = {
        'total_errors': len(errors),
        'invalid_responses': len([e for e in errors if e['predicted'] == -1]),
        'avg_true_count': np.mean(true_counts),
        'avg_predicted_count': np.mean(predicted_counts) if predicted_counts else -1,
        'underestimation_rate': len([e for e in errors if e['predicted'] < e['true'] and e['predicted'] != -1]) / len(errors),
        'overestimation_rate': len([e for e in errors if e['predicted'] > e['true']]) / len(errors),
    }
    
    return analysis


### Token functions

def find_list_token_positions(tokenizer, prompt: str, word_list: List[str]) -> Tuple[int, int]:
    """
    Find the start and end token positions of the word list in the prompt.
    Returns (start_pos, end_pos) where start_pos is the first word token, end_pos is after last word.
    """
    tokens = tokenizer.encode(prompt, return_tensors="pt")[0]
    token_strings = [tokenizer.decode([t]) for t in tokens]
    
    # Find the list start (look for '[')
    list_start_idx = None
    for i, token_str in enumerate(token_strings):
        if '[' in token_str:
            list_start_idx = i
            break
    
    if list_start_idx is None:
        raise ValueError("Could not find list start '[' in prompt")
    
    # Find first actual word after '['
    first_word_idx = None
    for i in range(list_start_idx + 1, len(token_strings)):
        token_str = token_strings[i].strip()
        if token_str and token_str not in ['[', ']', ' ', ',']:
            first_word_idx = i
            break
    
    # Find list end (look for ']')
    list_end_idx = None
    for i in range(first_word_idx, len(token_strings)):
        if ']' in token_strings[i]:
            list_end_idx = i + 1  # Include the ']'
            break
    
    if first_word_idx is None or list_end_idx is None:
        raise ValueError("Could not find word list boundaries")
    
    return first_word_idx, list_end_idx

def get_token_positions(tokenizer, prompt: str, word_list: List[str], option: str = "list_and_after"):
    """
    Get token positions to patch based on the option.
    
    Args:
        option: "list_and_after" (from first word to end) or "all" (all tokens)
    """
    tokens = tokenizer.encode(prompt, return_tensors="pt")[0]
    
    if option == "all":
        return list(range(len(tokens)))
    elif option == "list_and_after":
        first_word_idx, _ = find_list_token_positions(tokenizer, prompt, word_list)
        return list(range(first_word_idx, len(tokens)))
    else:
        raise ValueError(f"Unknown option: {option}")