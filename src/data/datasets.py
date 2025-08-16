"""Dataset classes for causal analysis."""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

@dataclass
class SentimentPoint:
    """Single sentiment analysis example."""
    prompt: str
    label: int  # 0=negative, 1=positive
    expected_continuation: str
    metadata: Optional[Dict] = None

class SentimentDataset(Dataset):
    """Dataset for sentiment analysis tasks."""
    
    def __init__(self, examples: List[SentimentPoint] = None):
        """Initialize sentiment dataset."""
        if examples is None:
            self.examples = self._create_default_dataset()
        else:
            self.examples = examples
    
    def _create_default_dataset(self) -> List[SentimentPoint]:
        """Create default sentiment dataset."""
        examples = []
        
        # Positive examples
        positive_prompts = [
            ("The movie was absolutely", " wonderful"),
            ("This product is", " good"),
            ("She felt incredibly", " happy"),
            ("The experience was", " amazing"),
        ]
        
        for prompt, continuation in positive_prompts:
            examples.append(SentimentPoint(
                prompt=prompt,
                label=1,
                expected_continuation=continuation,
                metadata={"category": "positive"}
            ))
        
        # Negative examples
        negative_prompts = [
            ("The movie was absolutely", " terrible"),
            ("This product is", " bad"),
            ("She felt incredibly", " sad"),
            ("The experience was", " awful"),
        ]
        
        for prompt, continuation in negative_prompts:
            examples.append(SentimentPoint(
                prompt=prompt,
                label=0,
                expected_continuation=continuation,
                metadata={"category": "negative"}
            ))
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> SentimentPoint:
        return self.examples[idx]
    
    def get_texts_and_labels(self) -> Tuple[List[str], List[int]]:
        """Get all texts and labels as lists."""
        texts = [ex.prompt for ex in self.examples]
        labels = [ex.label for ex in self.examples]
        return texts, labels
    
    def create_distribution_shifts(self) -> Dict[str, "SentimentDataset"]:
        """Create datasets with distribution shifts."""
        distributions = {}
        
        # Formal language
        formal_examples = [
            SentimentPoint("The presentation was", 1, " excellent"),
            SentimentPoint("The presentation was", 0, " poor"),
        ]
        distributions["formal"] = SentimentDataset(formal_examples)
        
        # Informal language
        informal_examples = [
            SentimentPoint("That party was super", 1, " fun"),
            SentimentPoint("That party was super", 0, " boring"),
        ]
        distributions["informal"] = SentimentDataset(informal_examples)
        
        return distributions