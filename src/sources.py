"""Code-snippet sources for the dataset builder.

Each loader is a generator yielding ``str`` (raw Python source snippets).
"""

from __future__ import annotations

from typing import Iterable, Iterator, Optional


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
    from datasets import load_dataset  # lazy import

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
    from datasets import load_dataset  # lazy import

    ds = load_dataset("code_search_net", language, split=split, streaming=True)
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        code = row.get("whole_func_string") or row.get("func_code_string")
        if code:
            yield code


SOURCES = {
    "demo": load_demo,
    "mbpp": load_mbpp,
    "code_search_net": load_code_search_net,
}


def iter_source(name: str, **kwargs) -> Iterable[str]:
    if name not in SOURCES:
        raise ValueError(f"unknown source {name!r}; choose from {sorted(SOURCES)}")
    return SOURCES[name](**kwargs)
