import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import gc

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

class ActivationCache:
    """Cache for storing activations during forward pass"""
    
    def __init__(self):
        self.cached_activations = {}
        self.hooks = []
    
    def clear(self):
        """Clear all cached activations and remove hooks"""
        self.cached_activations.clear()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def cache_residual_hook(self, layer_idx):
        """Create hook to cache residual stream after layer layer_idx"""
        def hook_fn(module, input, output):
            # For Llama, output is a tuple (hidden_states, ...)
            hidden_states = output[0] if isinstance(output, tuple) else output
            self.cached_activations[f'residual_{layer_idx}'] = hidden_states.clone()
        return hook_fn
    
    def cache_attn_hook(self, layer_idx):
        """Create hook to cache attention output"""
        def hook_fn(module, input, output):
            # output[0] is the attention output
            attn_output = output[0] if isinstance(output, tuple) else output
            self.cached_activations[f'attn_{layer_idx}'] = attn_output.clone()
        return hook_fn
    
    def cache_mlp_hook(self, layer_idx):
        """Create hook to cache MLP output"""
        def hook_fn(module, input, output):
            self.cached_activations[f'mlp_{layer_idx}'] = output.clone()
        return hook_fn

def cache_clean_activations(model, tokenizer, clean_prompt: str, word_list: List[str], 
                          token_option: str = "list_and_after") -> Dict:
    """
    Run clean prompt and cache activations.
    Returns dict with cached activations and metadata.
    """
    cache = ActivationCache()
    
    # Register hooks for all layers
    for layer_idx in range(len(model.model.layers)):
        layer = model.model.layers[layer_idx]
        
        # Cache residual stream (after layer norm)
        hook = layer.post_attention_layernorm.register_forward_hook(
            cache.cache_residual_hook(layer_idx)
        )
        cache.hooks.append(hook)
        
        # Cache attention output
        hook = layer.self_attn.register_forward_hook(
            cache.cache_attn_hook(layer_idx)
        )
        cache.hooks.append(hook)
        
        # Cache MLP output  
        hook = layer.mlp.register_forward_hook(
            cache.cache_mlp_hook(layer_idx)
        )
        cache.hooks.append(hook)
    
    # Run forward pass
    inputs = tokenizer(clean_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits and token positions
    logits = outputs.logits[0, -1, :]  # Last token logits
    token_positions = get_token_positions(tokenizer, clean_prompt, word_list, token_option)
    
    result = {
        'cached_activations': dict(cache.cached_activations),
        'logits': logits,
        'token_positions': token_positions,
        'input_ids': inputs['input_ids']
    }
    
    # Clean up
    cache.clear()
    
    return result

class ActivationPatcher:
    """Handles activation patching during forward pass"""
    
    def __init__(self, cached_activations: Dict, token_position: int):
        self.cached_activations = cached_activations
        self.token_position = token_position  # Single position now
        self.hooks = []
        self.patch_component = None
        self.patch_layer = None
    
    def clear(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def patch_residual_hook(self, layer_idx):
        """Create hook to patch residual stream at single token position"""
        def hook_fn(module, input, output):
            if self.patch_component == 'residual' and self.patch_layer == layer_idx:
                # Get cached activation
                cached_key = f'residual_{layer_idx}'
                if cached_key in self.cached_activations:
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    cached_activation = self.cached_activations[cached_key]
                    
                    # Patch single token position
                    hidden_states[:, self.token_position, :] = cached_activation[:, self.token_position, :]
                    
                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    else:
                        return hidden_states
            return output
        return hook_fn
    
    def patch_attn_hook(self, layer_idx):
        """Create hook to patch attention output at single token position"""
        def hook_fn(module, input, output):
            if self.patch_component == 'attn' and self.patch_layer == layer_idx:
                cached_key = f'attn_{layer_idx}'
                if cached_key in self.cached_activations:
                    attn_output = output[0] if isinstance(output, tuple) else output
                    cached_activation = self.cached_activations[cached_key]
                    
                    # Patch single token position
                    attn_output[:, self.token_position, :] = cached_activation[:, self.token_position, :]
                    
                    if isinstance(output, tuple):
                        return (attn_output,) + output[1:]
                    else:
                        return attn_output
            return output
        return hook_fn
    
    def patch_mlp_hook(self, layer_idx):
        """Create hook to patch MLP output at single token position"""
        def hook_fn(module, input, output):
            if self.patch_component == 'mlp' and self.patch_layer == layer_idx:
                cached_key = f'mlp_{layer_idx}'
                if cached_key in self.cached_activations:
                    cached_activation = self.cached_activations[cached_key]
                    
                    # Patch single token position
                    output[:, self.token_position, :] = cached_activation[:, self.token_position, :]
            return output
        return hook_fn

def setup_patching_hooks(model, patcher: ActivationPatcher):
    """Register all patching hooks"""
    for layer_idx in range(len(model.model.layers)):
        layer = model.model.layers[layer_idx]
        
        # Residual stream hook
        # hook = layer.post_attention_layernorm.register_forward_hook(
        #     patcher.patch_residual_hook(layer_idx)
        # )
        hook = layer.register_forward_hook(
            patcher.patch_residual_hook(layer_idx)
        )
        patcher.hooks.append(hook)
        
        # Attention hook
        hook = layer.self_attn.register_forward_hook(
            patcher.patch_attn_hook(layer_idx)
        )
        patcher.hooks.append(hook)
        
        # MLP hook
        hook = layer.mlp.register_forward_hook(
            patcher.patch_mlp_hook(layer_idx)
        )
        patcher.hooks.append(hook)

def run_patched_forward(model, tokenizer, corrupted_prompt: str, 
                       patch_component: str, patch_layer: int,
                       cached_activations: Dict, token_position: int) -> torch.Tensor:
    """
    Run forward pass with patching at specific component, layer, and token position.
    
    Args:
        patch_component: 'residual', 'attn', or 'mlp'
        patch_layer: layer index to patch
        token_position: single token position to patch
    """
    patcher = ActivationPatcher(cached_activations, token_position)
    patcher.patch_component = patch_component
    patcher.patch_layer = patch_layer
    
    # Setup hooks
    setup_patching_hooks(model, patcher)
    
    try:
        # Run forward pass
        inputs = tokenizer(corrupted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits[0, -1, :]  # Last token logits
        
    finally:
        # Always clean up hooks
        patcher.clear()
    
    return logits

def get_logit_difference(logits: torch.Tensor, tokenizer, 
                        correct_answer: str, incorrect_answer: str) -> float:
    """
    Calculate logit difference between correct and incorrect answers.
    Following paper's recommendation to use logit difference over probability.
    """
    # Get token IDs for answers
    correct_token_id = tokenizer.encode(correct_answer, add_special_tokens=False)[0]
    incorrect_token_id = tokenizer.encode(incorrect_answer, add_special_tokens=False)[0]
    
    # Calculate logit difference
    logit_diff = logits[correct_token_id] - logits[incorrect_token_id]
    
    return logit_diff.item()

def run_single_patch_experiment(model, tokenizer, clean_cache, clean_logit_diff, corrupted_logit_diff,
                                sample: Dict, patch_component: str, patch_layer: int, token_position: int) -> float:
    """
    Run single patching experiment for one component at one layer at one token position.
    Returns the patching effect (normalized).
    """
    clean_prompt = sample['clean_prompt']
    corrupted_prompt = sample['corrupted_prompt']
    clean_answer = str(sample['clean_target_count'])
    corrupted_answer = str(sample['corrupted_target_count'])
    word_list = sample['clean_word_list']
    
    # Check if token position is valid
    if token_position >= len(clean_cache['input_ids'][0]):
        return 0.0
    
    # Run patched forward pass
    patched_logits = run_patched_forward(
        model, tokenizer, corrupted_prompt,
        patch_component, patch_layer,
        clean_cache['cached_activations'], 
        token_position
    )
    
    # Calculate patching effect
    patched_logit_diff = get_logit_difference(patched_logits, tokenizer, clean_answer, corrupted_answer)
        
    # Normalize patching effect: (patched - corrupted) / (clean - corrupted)
    # This gives us a value between 0 and 1, where 1 means full recovery
    baseline_diff = clean_logit_diff - corrupted_logit_diff
    
    if baseline_diff != 0:
        patching_effect = (patched_logit_diff - corrupted_logit_diff) / baseline_diff
    else:
        print("logit diff between clean and corrupted is zero!")
        patching_effect = 0.0
    
    return patching_effect

def get_token_info(tokenizer, sample: Dict) -> Dict:
    """
    Get detailed token information for analysis.
    """
    clean_prompt = sample['clean_prompt']
    word_list = sample['clean_word_list']
    
    # Tokenize
    tokens = tokenizer.encode(clean_prompt, return_tensors="pt")[0]
    token_strings = [tokenizer.decode([t]) for t in tokens]
    
    # Find important positions
    first_word_idx, list_end_idx = find_list_token_positions(tokenizer, clean_prompt, word_list)
    
    # Create token info
    token_info = {
        'total_tokens': len(tokens),
        'token_strings': token_strings,
        'first_word_position': first_word_idx,
        'list_end_position': list_end_idx,
        'word_positions': [],  # Positions of each word in the list
        'list_positions': list(range(first_word_idx, list_end_idx))  # All positions in the list
    }
    
    # Find each word's position (approximate)
    for word in word_list:
        for i, token_str in enumerate(token_strings):
            if word.lower() in token_str.lower() and i >= first_word_idx and i < list_end_idx:
                token_info['word_positions'].append(i)
                break
    
    return token_info

def run_systematic_patching_by_position(model, tokenizer, sample: Dict, 
                                      max_layers: Optional[int] = None,
                                      position_option: str = "list_only") -> Dict:
    """
    Run systematic patching across all components, layers, and token positions for one sample.
    
    Args:
        position_option: 
            - "list_only": Test only positions within the word list
            - "list_and_after": Test list positions and everything after
            - "all": Test all token positions
            - "key_positions": Test only word positions and list end
    """
    if max_layers is None:
        max_layers = len(model.model.layers)
    
    # Get token information
    token_info = get_token_info(tokenizer, sample)

    # Get corruption position
    corruption_position = sample['corrupted_position']
    
    # Determine which positions to test    
    if position_option == "list_only":
        test_positions = token_info['list_positions']
    elif position_option == "list_and_after":
        test_positions = list(range(token_info['first_word_position'], token_info['total_tokens']))
    elif position_option == "all":
        test_positions = list(range(token_info['total_tokens']))
    elif position_option == "key_positions":
        test_positions = token_info['word_positions'] + [token_info['list_end_position'] - 1]
        test_positions = sorted(list(set(test_positions)))  # Remove duplicates
    else:
        raise ValueError(f"Unknown position_option: {position_option}")
    
    results = {
        'residual': {},
        'attn': {},
        'mlp': {},
        'token_info': token_info,
        'test_positions': test_positions,
        'sample_info': {
            'category': sample['category'],
            'clean_count': sample['clean_target_count'],
            'corrupted_count': sample['corrupted_target_count'],
            'example_id': sample['example_id']
        }
    }
    
    print(f"Processing sample {sample['example_id']} ({sample['category']})") 
    print(f"Clean: {sample['clean_target_count']}, Corrupted: {sample['corrupted_target_count']}")
    print(f"Testing {len(test_positions)} token positions across {max_layers} layers")
    
    # Test each component, layer, and position
    for component in ['residual', 'attn', 'mlp']:
    # for component in ['residual']:
        print(f"  Testing {component}...")
        results[component] = {}
        
        for layer in tqdm(range(max_layers), desc=f"{component} layers", leave=False):
            results[component][layer] = {}
            
            for i, pos in enumerate(test_positions):
                if i < corruption_position:
                    results[component][layer][pos] = 0.0
                    continue
                    
                # Clean run with caching
                clean_cache = cache_clean_activations(model, tokenizer, sample['clean_prompt'], sample['clean_word_list'], "all")

                # Corrupted run
                inputs_corrupted = tokenizer(sample['corrupted_prompt'], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs_corrupted = model(**inputs_corrupted)
                corrupted_logits = outputs_corrupted.logits[0, -1, :]
                
                # Calculate logit differences
                clean_answer = str(sample['clean_target_count'])
                corrupted_answer = str(sample['corrupted_target_count'])
                clean_logit_diff = get_logit_difference(clean_cache['logits'], tokenizer, clean_answer, corrupted_answer)
                corrupted_logit_diff = get_logit_difference(corrupted_logits, tokenizer, clean_answer, corrupted_answer)
                                    
                effect = run_single_patch_experiment(
                    model, tokenizer, clean_cache, clean_logit_diff, corrupted_logit_diff,
                    sample, component, layer, pos
                )
                results[component][layer][pos] = effect
        
            # Clear GPU cache periodically
            if layer % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    return results