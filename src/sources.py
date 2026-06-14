"""Code-snippet sources for the dataset builder.

Each loader is a generator yielding ``str`` (raw Python source snippets).
"""

from __future__ import annotations

from typing import Iterable, Iterator, Optional
from datasets import load_dataset
import re


_KNOWN_SPLIT_SIZES = {
    "demo": {"train": 8},
    "mbpp": {"train": 500, "validation": 90, "test": 374},
    "magicoder": {"train": 23284, "validation": 5000, "test": 10000},
    "codealpaca": {"train": 1336, "validation": 300, "test": 1000},
}


_DEMO_SNIPPETS = [
    # 1
    """def factorial(number):
    result = 1
    for value in range(2, number + 1):
        result = result * value
    return result
""",
    # 2
    """def is_prime(candidate):
    if candidate < 2:
        return False
    for divisor in range(2, int(candidate ** 0.5) + 1):
        if candidate % divisor == 0:
            return False
    return True
""",
    # 3
    """def fibonacci(limit):
    sequence = [0, 1]
    while sequence[-1] + sequence[-2] < limit:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence
""",
    # 4
    """class Counter:
    def __init__(self):
        self.total = 0

    def increment(self, amount=1):
        self.total += amount

    def reset(self):
        self.total = 0
""",
    # 5
    """def average(values):
    if not values:
        return 0.0
    return sum(values) / len(values)
""",
    # 6
    """def reverse_string(text):
    reversed_text = ''
    for character in text:
        reversed_text = character + reversed_text
    return reversed_text
""",
    # 7
    """def count_vowels(sentence):
    vowels = 'aeiou'
    total = 0
    for letter in sentence.lower():
        if letter in vowels:
            total += 1
    return total
""",
    # 8
    """def merge_sorted(left_list, right_list):
    merged = []
    i, j = 0, 0
    while i < len(left_list) and j < len(right_list):
        if left_list[i] <= right_list[j]:
            merged.append(left_list[i])
            i += 1
        else:
            merged.append(right_list[j])
            j += 1
    merged.extend(left_list[i:])
    merged.extend(right_list[j:])
    return merged
""",
]


def load_demo(max_samples: Optional[int] = None) -> Iterator[str]:
    """Bundled snippets — no network required."""
    snippets = _DEMO_SNIPPETS
    if max_samples is not None:
        snippets = snippets[:max_samples]
    yield from snippets


def load_mbpp(split: str = "train", max_samples: Optional[int] = None) -> Iterator[str]:
    """Mostly Basic Python Problems — small, clean, well-suited as a starting point."""

    ds = load_dataset("mbpp", split=split)
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        code = row.get("code")
        if code:
            yield code


def load_code_search_net(
    split: str = "train",
    max_samples: Optional[int] = None,
    language: str = "python",
) -> Iterator[str]:
    """CodeSearchNet — much larger; stream to avoid loading everything into memory."""

    ds = load_dataset("code_search_net", language, split=split, streaming=True)
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        code = row.get("whole_func_string") or row.get("func_code_string")
        if code:
            yield code


def load_magicoder(
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Iterator[str]:
    """Magicoder-OSS-Instruct-75K — large dataset with problem/solution pairs.
    
    Loads all Python samples from the dataset and splits them according to
    _KNOWN_SPLIT_SIZES configuration. The original dataset only has a 'train' split,
    but we redistribute the Python samples into train/validation/test splits.
    
    Filters for Python samples and extracts code from the 'solution' field.
    The solution field contains markdown code blocks that need to be extracted.
    """

    # Load all Python samples from the train split
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    
    # Extract all Python code samples
    all_samples = []
    for row in ds:
        # Only process Python samples
        if row.get("lang") != "python":
            continue
        # Extract code from solution field (contains markdown code blocks)
        solution = row.get("solution", "")
        # Extract code from markdown code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', solution, re.DOTALL)
        if code_blocks:
            # Join multiple code blocks if present
            code = "\n\n".join(code_blocks)
            all_samples.append(code)
    
    # Calculate split indices based on _KNOWN_SPLIT_SIZES
    split_sizes = _KNOWN_SPLIT_SIZES.get("magicoder", {})
    train_size = split_sizes.get("train", 0)
    val_size = split_sizes.get("validation", 0)
    test_size = split_sizes.get("test", 0)
    
    # Calculate split boundaries
    train_end = train_size
    val_end = train_size + val_size
    test_end = train_size + val_size + test_size
    
    # Select the appropriate split
    if split == "train":
        start, end = 0, train_end
    elif split == "validation":
        start, end = train_end, val_end
    elif split == "test":
        start, end = val_end, test_end
    else:
        return
    
    # Apply max_samples limit if specified
    if max_samples is not None:
        end = min(end, max_samples)
    
    # Yield samples from the selected split
    for i in range(start, min(end, len(all_samples))):
        yield all_samples[i]


def load_codealpaca(
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Iterator[str]:
    """CodeAlpaca-20k — dataset of code instruction-response pairs.
    
    Loads all Python-related samples from the dataset and splits them according to
    _KNOWN_SPLIT_SIZES configuration. The original dataset only has a 'train' split,
    but we redistribute the Python samples into train/validation/test splits.
    
    Filters for Python-related samples and extracts code from the 'output' field.
    """
    # Load all samples from the train split
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    
    # Extract Python-related code samples
    all_samples = []
    for row in ds:
        instruction = row.get("instruction", "").lower()
        output = row.get("output", "")
        
        # Filter for Python-related samples
        if "python" not in instruction and "python" not in output.lower():
            continue
        
        # Extract code from output field
        code = output.strip()
        if code and len(code) > 10:  # Filter out very short outputs
            all_samples.append(code)
    
    # Calculate split indices based on _KNOWN_SPLIT_SIZES
    split_sizes = _KNOWN_SPLIT_SIZES.get("codealpaca", {})
    train_size = split_sizes.get("train", 0)
    val_size = split_sizes.get("validation", 0)
    test_size = split_sizes.get("test", 0)
    
    # Calculate split boundaries
    train_end = train_size
    val_end = train_size + val_size
    test_end = train_size + val_size + test_size
    
    # Select the appropriate split
    if split == "train":
        start, end = 0, train_end
    elif split == "validation":
        start, end = train_end, val_end
    elif split == "test":
        start, end = val_end, test_end
    else:
        return
    
    # Apply max_samples limit if specified
    if max_samples is not None:
        end = min(end, max_samples)
    
    # Yield samples from the selected split
    for i in range(start, min(end, len(all_samples))):
        yield all_samples[i]


SOURCES = {
    "demo": load_demo,
    "mbpp": load_mbpp,
    "code_search_net": load_code_search_net,
    "magicoder": load_magicoder,
    "codealpaca": load_codealpaca,
}


def iter_source(name: str, **kwargs) -> Iterable[str]:
    if name not in SOURCES:
        raise ValueError(f"unknown source {name!r}; choose from {sorted(SOURCES)}")
    return SOURCES[name](**kwargs)
