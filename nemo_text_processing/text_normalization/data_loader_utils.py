# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import json
import re
import string
import sys
import unicodedata
from collections import defaultdict, namedtuple
from typing import Dict, List, Optional, Set, Tuple
from unicodedata import category

from nemo_text_processing.utils.logging import logger

NFC = 'NFC'
EOS_TYPE = "EOS"
PUNCT_TYPE = "PUNCT"
PLAIN_TYPE = "PLAIN"
Instance = namedtuple('Instance', 'token_type un_normalized normalized')
known_types = [
    "PLAIN",
    "DATE",
    "CARDINAL",
    "LETTERS",
    "VERBATIM",
    "MEASURE",
    "DECIMAL",
    "ORDINAL",
    "DIGIT",
    "MONEY",
    "TELEPHONE",
    "ELECTRONIC",
    "FRACTION",
    "TIME",
    "ADDRESS",
    "ROMAN",
    "RANGE",
]


def _load_kaggle_text_norm_file(file_path: str, to_lower: bool) -> List[Instance]:
    """
    https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish
    Loads text file in the Kaggle Google text normalization file format: <semiotic class>\t<unnormalized text>\t<`self` if trivial class or normalized text>
    E.g.
    PLAIN   Brillantaisia   <self>
    PLAIN   is      <self>
    PLAIN   a       <self>
    PLAIN   genus   <self>
    PLAIN   of      <self>
    PLAIN   plant   <self>
    PLAIN   in      <self>
    PLAIN   family  <self>
    PLAIN   Acanthaceae     <self>
    PUNCT   .       sil
    <eos>   <eos>

    Args:
        file_path: file path to text file

    Returns: flat list of instances
    """
    pass


def load_files(file_paths: List[str], load_func=_load_kaggle_text_norm_file, to_lower: bool = True) -> List[Instance]:
    """
    Load given list of text files using the `load_func` function.

    Args:
        file_paths: list of file paths
        load_func: loading function

    Returns: flat list of instances
    """
    pass


def clean_generic(text: str) -> str:
    """
    Cleans text without affecting semiotic classes.

    Args:
        text: string

    Returns: cleaned string
    """
    pass


def evaluate(preds: List[str], labels: List[str], input: Optional[List[str]] = None, verbose: bool = True) -> float:
    """
    Evaluates accuracy given predictions and labels.

    Args:
        preds: predictions
        labels: labels
        input: optional, only needed for verbosity
        verbose: if true prints [input], golden labels and predictions

    Returns accuracy
    """
    pass


def training_data_to_tokens(
    data: List[Instance], category: Optional[str] = None
) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Filters the instance list by category if provided and converts it into a map from token type to list of un_normalized and normalized strings

    Args:
        data: list of instances
        category: optional semiotic class category name

    Returns Dict: token type -> (list of un_normalized strings, list of normalized strings)
    """
    pass


def training_data_to_sentences(data: List[Instance]) -> Tuple[List[str], List[str], List[Set[str]]]:
    """
    Takes instance list, creates list of sentences split by EOS_Token
    Args:
        data: list of instances
    Returns (list of unnormalized sentences, list of normalized sentences, list of sets of categories in a sentence)
    """
    pass


def post_process_punctuation(text: str) -> str:
    """
    Normalized quotes and spaces

    Args:
        text: text

    Returns: text with normalized spaces and quotes
    """
    pass


def pre_process(text: str) -> str:
    """
    Optional text preprocessing before normalization (part of TTS TN pipeline)

    Args:
        text: string that may include semiotic classes

    Returns: text with spaces around punctuation marks
    """
    pass


def load_file(file_path: str) -> List[str]:
    """
    Loads given text file with separate lines into list of string.

    Args:
        file_path: file path

    Returns: flat list of string
    """
    pass


def write_file(file_path: str, data: List[str]):
    """
    Writes out list of string to file.

    Args:
        file_path: file path
        data: list of string

    """
    pass


def post_process_punct(input: str, normalized_text: str, add_unicode_punct: bool = False):
    """
    Post-processing of the normalized output to match input in terms of spaces around punctuation marks.
    After NN normalization, Moses detokenization puts a space after
    punctuation marks, and attaches an opening quote "'" to the word to the right.
    E.g., input to the TN NN model is "12 test' example",
    after normalization and detokenization -> "twelve test 'example" (the quote is considered to be an opening quote,
    but it doesn't match the input and can cause issues during TTS voice generation.)
    The current function will match the punctuation and spaces of the normalized text with the input sequence.
    "12 test' example" -> "twelve test 'example" -> "twelve test' example" (the quote was shifted to match the input).

    Args:
        input: input text (original input to the NN, before normalization or tokenization)
        normalized_text: output text (output of the TN NN model)
        add_unicode_punct: set to True to handle unicode punctuation marks as well as default string.punctuation (increases post processing time)
    """
    pass
