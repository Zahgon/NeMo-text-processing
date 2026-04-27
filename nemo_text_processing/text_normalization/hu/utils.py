# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan.
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

import csv
import os


def get_abs_path(rel_path):
    """
    Get absolute path

    Args:
        rel_path: relative path to this file

    Returns absolute path
    """
    return os.path.dirname(os.path.abspath(__file__)) + '/' + rel_path


def load_labels(abs_path):
    """
    loads relative path file as dictionary

    Args:
        abs_path: absolute path

    Returns dictionary of mappings
    """
    with open(abs_path) as label_tsv:
        labels = list(csv.reader(label_tsv, delimiter="\t"))
        return labels


def load_inflection(abs_path):
    """
    loads inflection information

    Args:
        abs_path: absolute path

    Returns dictionary of mappings of word endings to
    lists of case endings.
    """
    pass


def _modify_ending(outword: str, word: str, form: str) -> str:
    """
    Helper for the inflector. Modifies endings where there is a difference
    between how they are written for abbreviations, and for full words.

    Args:
        outword: the form of the word to be output
        word: the base form of the word
        form: the ending to be appended
    """
    pass


def inflect_abbreviation(abbr: str, word: str, singular_only=False):
    """
    For currency symbols, the inflection can either be taken from
    the underlying final word, or from the letter itself.
    This (ab)uses naive_inflector to get the letter-based
    inflection.

    Args:
        abbr: the abbreviated base form
        word: the base (nominative singular) form of the expansion
              of abbr
        singular_only: whether or not to add plural forms

    Returns a list of tuples containing the inflected abbreviation and
    its expansion.
    """
    pass


def naive_inflector(abbr: str, word: str, singular_only=False):
    """
    Performs naÃ¯ve inflection of a pair of words: the abbreviation,
    and its expansion. Possessive forms are omitted, due to the
    nature of the kinds of words/abbreviations being expanded

    Args:
        abbr: the abbreviated base form
        word: the base (nominative singular) form of the expansion
              of abbr
        singular_only: whether or not to add plural forms

    Returns a list of tuples containing the inflected abbreviation and
    its expansion.
    """
    pass
