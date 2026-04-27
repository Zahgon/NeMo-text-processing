# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os

####################
# HEBREW CONSTANTS #
####################
units_feminine_dict = {
    "0": "×�×¤×¡",
    "1": "×�×—×ª",
    "2": "×©×ª×™×™×�",
    "3": "×©×œ×•×©",
    "4": "×�×¨×‘×¢",
    "5": "×—×ž×©",
    "6": "×©×©",
    "7": "×©×‘×¢",
    "8": "×©×ž×•× ×”",
    "9": "×ª×©×¢",
}

units_masculine_dict = {
    "0": "×�×¤×¡",
    "1": "×�×—×“",
    "2": "×©× ×™×™×�",
    "3": "×©×œ×•×©×”",
    "4": "×�×¨×‘×¢×”",
    "5": "×—×ž×™×©×”",
    "6": "×©×™×©×”",
    "7": "×©×‘×¢×”",
    "8": "×©×ž×•× ×”",
    "9": "×ª×©×¢×”",
}

tens_dict = {
    "2": "×¢×©×¨×™×�",
    "3": "×©×œ×•×©×™×�",
    "4": "×�×¨×‘×¢×™×�",
    "5": "×—×ž×™×©×™×�",
    "6": "×©×™×©×™×�",
    "7": "×©×‘×¢×™×�",
    "8": "×©×ž×•× ×™×�",
    "9": "×ª×©×¢×™×�",
}

ten = {
    "short": "×¢×©×¨",
    "long": "×¢×©×¨×”",
}  # double pronunciation: short is 'eser' and 'asar', long is 'esre' and 'asara'


#############
# FUNCTIONS #
#############
def get_abs_path(rel_path):
    """
    Get absolute path

    Args:
        rel_path: relative path to this file

    Returns absolute path
    """
    return os.path.dirname(os.path.abspath(__file__)) + "/" + rel_path


def augment_labels_with_punct_at_end(labels):
    """
    augments labels: if key ends on a punctuation that value does not have, add a new label
    where the value maintains the punctuation

    Args:
        labels : input labels
    Returns:
        additional labels
    """
    res = []
    for label in labels:
        if len(label) > 1:
            if label[0][-1] == "." and label[1][-1] != ".":
                res.append([label[0], label[1] + "."] + label[2:])
    return res


def digit_by_digit(num):

    pass


def integer_to_text(num, only_fem=False):
    pass


def _less_than_10(num, only_fem=False):
    """
    Returns a list of all the possible names of a number in range 0-9
    """
    pass


def _less_than_100(num, only_fem=False):
    """
    Returns a list of all the possible names of a number in range 0-99
    """
    pass
