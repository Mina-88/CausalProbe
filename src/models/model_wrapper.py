"""Model wrapper for unified interface."""

import torch
from typing import Dict, List, Tuple, Optional, Callable
from transformer_lens import HookedTransformer
import logging

logger = logging.getLogger(__name__)

class ModelWrapper:
    """Wrapper for transformer models with intervention capabilities."""
    
    def __init__(self, model_name: str = "gpt2-small"):
        """Initialize model wrapper."""
        self.model_name = model_name
        self.model = self._load_model()
        self.hooks = {}
        self._cache = {}
        
    def _load_model(self) -> HookedTransformer:
        """Load and configure model."""
        logger.info(f"Loading model: {self.model_name}")
        model = HookedTransformer.from_pretrained(
            self.model_name,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            refactor_factored_attn_matrices=False,
        )
        model.set_use_attn_result(True)
        model.set_use_split_qkv_input(True)
        return model
    
    def get_activations(self, 
                       texts: List[str], 
                       layer_idx: int,
                       use_cache: bool = True) -> torch.Tensor:
        """Extract activations at specified layer."""
        cache_key = f"{tuple(texts)}_{layer_idx}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        _, cache = self.model.run_with_cache(texts)
        hook_name = f"blocks.{layer_idx}.mlp.hook_post"
        activations = cache[hook_name][:, -1, :]  # Last token
        
        if use_cache:
            self._cache[cache_key] = activations
        
        return activations
    
    def add_intervention_hook(self,
                             layer_idx: int,
                             intervention_fn: Callable,
                             hook_type: str = "mlp") -> str:
        """Add intervention hook to model."""
        if hook_type == "mlp":
            hook_name = f"blocks.{layer_idx}.mlp.hook_post"
        elif hook_type == "attn":
            hook_name = f"blocks.{layer_idx}.attn.hook_result"
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")
        
        self.model.add_hook(hook_name, intervention_fn)
        self.hooks[hook_name] = intervention_fn
        return hook_name
    
    def reset_hooks(self):
        """Remove all hooks."""
        self.model.reset_hooks()
        self.hooks = {}
    
    def predict(self, text: str) -> Tuple[str, torch.Tensor]:
        """Get model prediction for text."""
        tokens = self.model.to_tokens(text)
        with torch.no_grad():
            logits = self.model(tokens)
        
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        top_token_id = torch.argmax(probs).item()
        top_token = self.model.to_string(top_token_id)
        
        return top_token, probs
    
    def clear_cache(self):
        """Clear activation cache."""
        self._cache = {}
        logger.info("Cleared activation cache")