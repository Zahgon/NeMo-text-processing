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


import argparse
import logging
import os
import pickle
import re
import shutil
from typing import Dict, List

import model_utils
import pandas as pd
import utils
from joblib import Parallel, delayed
from tqdm import tqdm

from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

parser = argparse.ArgumentParser(description="Re-scoring")
parser.add_argument("--lang", default="en", type=str, choices=["en"])
parser.add_argument("--n_tagged", default=100, type=int, help="Number WFST options")
parser.add_argument("--context_len", default=-1, type=int, help="Context length, -1 to use full context")
parser.add_argument("--threshold", default=0.2, type=float, help="delta threshold value")
parser.add_argument("--overwrite_cache", action="store_true", help="overwrite cache")
parser.add_argument("--model_name", type=str, default="bert-base-uncased")
parser.add_argument("--cache_dir", default='cache', type=str, help="use cache dir")
parser.add_argument(
    "--data",
    default="text_normalization_dataset_files/EngConf.txt",
    help="For En only. Path to a file for evaluation.",
)
parser.add_argument("--n_jobs", default=-2, type=int, help="The maximum number of concurrently running jobs")
parser.add_argument(
    "--models", default="mlm_bert-base-uncased", type=str, help="Comma separated string of model names"
)
parser.add_argument(
    "--regenerate_pkl",
    action="store_true",
    help="Set to True to re-create pickle file with WFST normalization options",
)
parser.add_argument("--batch_size", default=200, type=int, help="Batch size for parallel processing")


def rank(sentences: List[str], labels: List[int], models: Dict[str, 'Model'], context_len=None, do_lower=True):
    """
    computes scores for each sentences using all provided models and returns summary in data frame
    """
    pass


def threshold_weights(norm_texts_weights, delta: float = 0.2):
    """
    norm_texts_weights: list of [ List[normalized options of input], list[weights] ]
    delta: delta to add to minimum weight in options to compose upper limit for threshhold

    returns:
        filter list of same format as input
    """
    pass


def _get_unchanged_count(text):
    """
    returns number of unchanged words in text
    """
    pass


def _get_replacement_count(text):
    """
    returns number of token replacements
    """
    pass


def threshold(norm_texts_weights, unchanged=True, replacement=True):
    """
    Reduces the number of WFST options based for LM rescoring.

    Args:
        :param norm_texts_weights: WFST options with associated weight
        :param unchanged: set to True to filter out examples based on number of words left unchanged
            (punct is not taken into account)
        :param replacement: set to True to filter out examples based on number of replacements made
            (Given A and B are WFST options, if the number of unchanged for A and B are the same,
            the option with a smaller number of replacements is preferable (i.e., larger span)).

    :return: WFST options with associated weight (reduced)
    """
    pass


def main():
    pass


if __name__ == "__main__":
    all_correct = main()
    print(f"all_correct: {all_correct}")
