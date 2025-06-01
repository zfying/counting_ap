from typing import List, Dict, Tuple, Optional
import torch
import utils


### Patching functions

class ActivationPatcher:
    """Handles activation patching during forward pass"""
    
    def __init__(self, cached_activations: Dict, token_positions: List[int]):
        self.cached_activations = cached_activations
        self.token_positions = token_positions
        self.hooks = []
        self.patch_component = None
        self.patch_layer = None
    
    def clear(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def patch_residual_hook(self, layer_idx):
        """Create hook to patch residual stream"""
        def hook_fn(module, input, output):
            if self.patch_component == 'residual' and self.patch_layer == layer_idx:
                # Get cached activation
                cached_key = f'residual_{layer_idx}'
                if cached_key in self.cached_activations:
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    cached_activation = self.cached_activations[cached_key]
                    
                    # Patch specified token positions
                    hidden_states[:, self.token_positions, :] = cached_activation[:, self.token_positions, :]
                    
                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    else:
                        return hidden_states
            return output
        return hook_fn
    
    def patch_attn_hook(self, layer_idx):
        """Create hook to patch attention output"""
        def hook_fn(module, input, output):
            if self.patch_component == 'attn' and self.patch_layer == layer_idx:
                cached_key = f'attn_{layer_idx}'
                if cached_key in self.cached_activations:
                    attn_output = output[0] if isinstance(output, tuple) else output
                    cached_activation = self.cached_activations[cached_key]
                    
                    # Patch specified token positions
                    attn_output[:, self.token_positions, :] = cached_activation[:, self.token_positions, :]
                    
                    if isinstance(output, tuple):
                        return (attn_output,) + output[1:]
                    else:
                        return attn_output
            return output
        return hook_fn
    
    def patch_mlp_hook(self, layer_idx):
        """Create hook to patch MLP output"""
        def hook_fn(module, input, output):
            if self.patch_component == 'mlp' and self.patch_layer == layer_idx:
                cached_key = f'mlp_{layer_idx}'
                if cached_key in self.cached_activations:
                    cached_activation = self.cached_activations[cached_key]
                    
                    # Patch specified token positions
                    output[:, self.token_positions, :] = cached_activation[:, self.token_positions, :]
            return output
        return hook_fn

def setup_patching_hooks(model, patcher: ActivationPatcher):
    """Register all patching hooks"""
    for layer_idx in range(len(model.model.layers)):
        layer = model.model.layers[layer_idx]
        
        # Residual stream hook
        hook = layer.post_attention_layernorm.register_forward_hook(
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
                       cached_activations: Dict, token_positions: List[int]) -> torch.Tensor:
    """
    Run forward pass with patching at specific component and layer.
    
    Args:
        patch_component: 'residual', 'attn', or 'mlp'
        patch_layer: layer index to patch
    """
    patcher = ActivationPatcher(cached_activations, token_positions)
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

def run_baseline_comparison(model, tokenizer, sample: Dict) -> Dict:
    """
    Run clean and corrupted baselines to establish logit differences.
    """
    clean_prompt = sample['clean_prompt']
    corrupted_prompt = sample['corrupted_prompt']
    clean_answer = str(sample['clean_target_count'])
    corrupted_answer = str(sample['clean_target_count'])
    
    # Run clean
    inputs_clean = tokenizer(clean_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs_clean = model(**inputs_clean)
    clean_logits = outputs_clean.logits[0, -1, :]
    
    # Run corrupted  
    inputs_corrupted = tokenizer(corrupted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs_corrupted = model(**inputs_corrupted)
    corrupted_logits = outputs_corrupted.logits[0, -1, :]
    
    # Calculate logit differences
    clean_logit_diff = get_logit_difference(clean_logits, tokenizer, clean_answer, corrupted_answer)
    corrupted_logit_diff = get_logit_difference(corrupted_logits, tokenizer, clean_answer, corrupted_answer)
    
    return {
        'clean_logit_diff': clean_logit_diff,
        'corrupted_logit_diff': corrupted_logit_diff,
        'baseline_diff': clean_logit_diff - corrupted_logit_diff  # How much corruption hurt performance
    }

### Caching functions

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
    token_positions = utils.get_token_positions(tokenizer, clean_prompt, word_list, token_option)
    
    result = {
        'cached_activations': dict(cache.cached_activations),
        'logits': logits,
        'token_positions': token_positions,
        'input_ids': inputs['input_ids']
    }
    
    # Clean up
    cache.clear()
    
    return result