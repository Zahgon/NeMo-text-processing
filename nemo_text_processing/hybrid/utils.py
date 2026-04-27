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


import copy
import difflib
import json
import logging
import re
import string
from typing import List, Optional, Tuple, Union

import pandas as pd
import pynini
from pynini.lib.rewrite import top_rewrite
from tqdm import tqdm

from nemo_text_processing.inverse_text_normalization.en.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

DELIMITER = '~~'

cardinal_graph = CardinalFst(input_case="cased").graph_no_exception
cardinal_graph = (
    pynini.closure(pynini.union("In ", "in ")) + cardinal_graph + pynini.closure(pynini.accep(" ") + cardinal_graph)
)

inverse_normalizer = InverseNormalizer()


def load_data(input_fs: List[str]):
    """
    loads data from list of abs file paths
    Returns:
        inputs: List[str] list of abs file paths
        targets: List[List[str]] list of targets, can contain multiple options for each target
        sentences: List[List[str]] list of sentence options
        labels: List[List[int]] list of labels (1,0)
    """
    pass


def remove_whitelist_boudaries(x):
    # remove raw whitelist
    pass


def _clean_pre_norm_libritts(inputs: List[str], targets: List[List[str]]):
    """
    standardizes format of inputs and targets before being normalized, so more rules apply.
    This is specific for libritts.
    """
    pass


def _clean_pre_norm_google(inputs: List[str], targets: List[List[str]]):
    """
    standardizes format of inputs and targets before being normalized, so more rules apply.
    This is specific for google dataset.
    """
    pass


def clean_pre_norm(inputs: List[str], targets: List[List[str]], dataset: Optional[str] = None):
    """
    standardizes format of inputs and targets before being normalized, so more rules apply.
    """
    pass


def _clean_post_norm_libritts(inputs: List[str], targets: List[List[str]], norm_texts):
    pass


def _clean_post_norm_google(inputs: List[str], targets: List[List[str]], norm_texts):
    """
    standardizes format of inputs and targets, and predicted normalizations for easier evaluation.
    This is specific for google dataset.
    """
    pass


def _clean_post_general(str) -> str:
    """
    standardizes format of inputs and targets, and predicted normalizations for easier evaluation.
    """
    pass


def _clean_targets(str) -> str:
    """Clean ground truth options."""
    pass


def adjust_pred(pred: str, gt: str, dataset: str, delim_present=True):
    """Standardize prediction format to make evaluation easier"""
    pass


def clean_post_norm(
    inputs: List[str],
    targets: List[List[str]],
    norm_texts,
    dataset: Optional[str] = None,
    delim_present: Optional[bool] = True,
):
    """
    Args:
        inputs (List[str]): inputs
        targets (List[List[str]]): targets
        norm_texts (List[(List[str], List[float])]): List of normalization options, weights
        dataset (Optional[str], optional): _description_. Defaults to None.
        delim_present (Optional[str], optional): The flag indicates whether normalization output contain delimiters "<>".
            Set to False for NN baseline.
    """
    pass


def clean_libri_tts(target: str):
    """
    Replace abbreviations in LibriTTS dataset
    """
    pass


def remove_punctuation(text: str, remove_spaces=True, do_lower=True, lang="en", exclude=None):
    """Removes punctuation (and optionally spaces) in text for better evaluation"""
    pass


def get_alternative_label(pred: str, targets: List[str]) -> bool:
    """Returns true if prediction matches target options"""
    pass


def get_labels(
    targets: List[str],
    norm_texts_weights: List[Tuple[str, str]],
    lang="en",
) -> List[List[str]]:
    """
    Assign labels to generated normalization options (1 - for ground truth, 0 - other options)
    Args:
        targets: ground truth normalization sentences
        norm_texts_weights: List of tuples: (normalization options, weights of normalization options)
    returns:
        List of labels [1, 0] for every normalization option
    """
    pass


def contains_month(pred, gt):
    """Check is the pred/gt contain month in the span"""
    pass


def is_date(pred, gt, cardinal_graph):
    """Returns True is pred and gt are date format modifications and are equal."""
    pass


def is_correct(pred: str, targets: Union[List[str], str], lang: str) -> bool:
    """
    returns True if prediction matches targets for language lang.
    """
    pass


def print_df(df):
    """
    prints data frame
    """
    pass


def get_diff(a: str, b: str):
    """returns list of different substrings between and b

    Returns:
        list of Tuple(pred start and end, gt start and end) subsections
    """
    pass


def diff_pred_gt(pred: str, gt: str):
    """returns list of different substrings between prediction and gt
    relies on that prediction uses '< '  ' >'

    Args:
        pred (str): prediction
        gt (str): ground truth

    Returns:
        list of Tuple(pred start and end, gt start and end) subsections

    e.g. pred="< Edward third >., king Our own . loss had been < two thousand two hundred >"
         gt  ="Edward III., king Our own loss had been twenty two hundred"
         --> [([0, 16], [0, 10]),      ([32, 34], [26, 26]),      ([48, 76], [40, 58])]
    """
    pass
