from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tqdm.std import tqdm as std_tqdm


def progress(iterable: Iterable[Any], **kwargs: Any):
    return std_tqdm(iterable, **kwargs)
