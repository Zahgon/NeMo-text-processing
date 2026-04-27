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
from pathlib import Path

import pynini
from pynini import Far
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, delete_space
from nemo_text_processing.text_normalization.en.utils import load_labels

NEMO_ALPHA_HE = pynini.union(*"×�×‘×’×“×”×•×–×—×˜×™×›×œ×ž×�× ×Ÿ×¡×¢×¤×£×¦×¥×§×¨×©×ª").optimize()
delete_and = pynutil.delete("×•")
delete_optional_and = delete_and.ques

####################
MIN_NEG_WEIGHT = -0.0001
MIN_POS_WEIGHT = 0.0001
MINUS = pynini.union("×ž×™× ×•×¡").optimize()


def string_map_cased(input_file: str):
    pass


def apply_fst(text, fst):
    """Given a string input, returns the output string
    produced by traversing the path with lowest weight.
    If no valid path accepts input string, returns an
    error.
    """
    pass


class GraphFst:
    """
    Base class for all grammar fsts.

    Args:
        name: name of grammar class
        kind: either 'classify' or 'verbalize'
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, name: str, kind: str, deterministic: bool = True):
        self.name = name
        self.kind = kind
        self._fst = None
        self.deterministic = deterministic

        self.far_path = Path(os.path.dirname(__file__) + "/grammars/" + kind + "/" + name + ".far")
        if self.far_exist():
            self._fst = Far(self.far_path, mode="r", arc_type="standard", far_type="default").get_fst()

    def far_exist(self) -> bool:
        """
        Returns true if FAR can be loaded
        """
        pass

    @property
    def fst(self) -> "pynini.FstLike":
        pass

    @fst.setter
    def fst(self, fst):
        pass

    def add_tokens(self, fst) -> "pynini.FstLike":
        """
        Wraps class name around to given fst

        Args:
            fst: input fst

        Returns:
            Fst: fst
        """
        pass

    def delete_tokens(self, fst) -> "pynini.FstLike":
        """
        Deletes class name wrap around output of given fst

        Args:
            fst: input fst

        Returns:
            Fst: fst
        """
        pass
