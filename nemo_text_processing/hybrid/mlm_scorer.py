# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2020 AWSLABS, AMAZON.
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

from typing import List

import numpy as np

try:
    import torch
    from torch.nn.functional import softmax
except ImportError as e:
    raise ImportError("torch is not installed")
try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except ImportError as e:
    raise ImportError("transformers is not installed")

__all__ = ['MLMScorer']


class MLMScorer:
    def __init__(self, model_name: str, device: str = 'cpu'):
        """
        Creates MLM scorer from https://arxiv.org/abs/1910.14659.
        Args:
            model_name: HuggingFace pretrained model name
            device: either 'cpu' or 'cuda'
        """
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.device = device
        self.MASK_LABEL = self.tokenizer.mask_token

    def score_sentences(self, sentences: List[str]):
        """
        returns list of MLM scores for each sentence in list.
        """
        pass

    def score_sentence(self, sentence: str):
        """
        returns MLM score for sentence.
        """
        pass

    def __mask_text__(self, idx: int, tokens: List[str]):
        """
        replaces string at index idx in list `tokens` with a masked token and returns the modified list.
        """
        masked = tokens.copy()
        masked[idx] = self.MASK_LABEL
        return masked
