# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
# Copyright (c) 2023, Jim O'Regan for SprÃ¥kbanken Tal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pynini
from pynini.lib import byte, pynutil

from nemo_text_processing.text_normalization.en.graph_utils import delete_space, insert_space

from .utils import get_abs_path, load_labels

_ALPHA_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZÃ…Ã„Ã–ÃœÃ‰"
_ALPHA_LOWER = "abcdefghijklmnopqrstuvwxyzÃ¥Ã¤Ã¶Ã¼Ã©"

TO_LOWER = pynini.union(*[pynini.cross(x, y) for x, y in zip(_ALPHA_UPPER, _ALPHA_LOWER)])
TO_UPPER = pynini.invert(TO_LOWER)

SV_LOWER = pynini.union(*_ALPHA_LOWER).optimize()
SV_UPPER = pynini.union(*_ALPHA_UPPER).optimize()
SV_ALPHA = pynini.union(SV_LOWER, SV_UPPER).optimize()
SV_ALNUM = pynini.union(byte.DIGIT, SV_ALPHA).optimize()

bos_or_space = pynini.union("[BOS]", " ")
eos_or_space = pynini.union("[EOS]", " ")

ensure_space = pynini.cross(pynini.closure(delete_space, 0, 1), " ")


def roman_to_int(fst: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Alters given fst to convert Roman integers (lower and upper cased) into Arabic numerals. Valid for values up to 1000.
    e.g.
        "V" -> "5"
        "i" -> "1"

    Args:
        fst: Any fst. Composes fst onto Roman conversion outputs.
    """
    pass
