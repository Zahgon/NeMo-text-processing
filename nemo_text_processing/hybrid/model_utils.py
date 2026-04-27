# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import logging
import math
import re
from typing import List, Union

from tqdm import tqdm

from nemo_text_processing.hybrid.mlm_scorer import MLMScorer

try:
    import torch
except ImportError as e:
    raise ImportError("torch is not installed")


def init_models(model_name_list: str):
    """
    returns dictionary of Masked Language Models by their HuggingFace name.
    """
    pass


def get_score(texts: Union[List[str], str], model: MLMScorer):
    """Computes MLM score for list of text using model"""
    pass


def get_masked_score(text, model, do_lower=True):
    """text is normalized prediction which contains <> around semiotic tokens.
    If multiple tokens are present, multiple variants of the text are created where all but one ambiguous semiotic tokens are masked
    to avoid unwanted reinforcement of neighboring semiotic tokens."""
    pass


def _get_ambiguous_positions(sentences: List[str]):
    """returns None or index list of ambigous semiotic tokens for list of sentences.
    E.g. if sentences = ["< street > < three > A", "< saint > < three > A"], it returns [1, 0] since only
    the first semiotic span <street>/<saint> is ambiguous."""
    pass


def score_options(sentences: List[str], context_len, model, do_lower=True):
    """return list of scores for each sentence in list where model is used for MLM Scoring."""
    pass


def find_diff(text, context_len=3):
    """Finds parts of text normalized by WFST and returns them in list with a context of context_len"""
    pass
