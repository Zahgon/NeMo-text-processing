# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_SPACE, GraphFst
from nemo_text_processing.text_normalization.vi.utils import get_abs_path, load_labels


class RomanFst(GraphFst):
    """
    Finite state transducer for classifying roman numbers in Vietnamese context:
        e.g. "tháº¿ ká»‰ XV" -> tokens { roman { key_cardinal: "tháº¿ ká»‰" integer: "mÆ°á»�i lÄƒm" } }
        e.g. "tháº¿ ká»· IV" -> tokens { roman { key_cardinal: "tháº¿ ká»·" integer: "bá»‘n" } }
        e.g. "thá»© IV" -> tokens { roman { key_cardinal: "thá»©" integer: "bá»‘n" } }
        e.g. "chÆ°Æ¡ng III" -> tokens { roman { key_cardinal: "chÆ°Æ¡ng" integer: "ba" } }
        e.g. "pháº§n ix" -> tokens { roman { key_cardinal: "pháº§n" integer: "chÃ­n" } }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="roman", kind="classify", deterministic=deterministic)

        key_words = []
        key_word_path = get_abs_path("data/roman/key_word.tsv")
        for k_word in load_labels(key_word_path):
            key_words.append(k_word[0])

        key_words_fst = pynini.union(*[pynini.accep(word) for word in key_words]).optimize()

        roman_numeral_path = get_abs_path("data/roman/roman_numerals.tsv")
        roman_numeral_pairs = load_labels(roman_numeral_path)

        roman_to_arabic = {}
        for roman, value in roman_numeral_pairs:
            roman_to_arabic[roman] = value
            roman_to_arabic[roman.lower()] = value

        self.arabic_to_roman = {}
        for roman, value in roman_numeral_pairs:
            self.arabic_to_roman[int(value)] = roman

        valid_roman_pairs = []
        for i in range(1, 4000):
            roman_upper = self._int_to_roman(i)
            roman_lower = roman_upper.lower()
            valid_roman_pairs.append((roman_upper, str(i)))
            valid_roman_pairs.append((roman_lower, str(i)))

        roman_to_arabic_fst = pynini.string_map(valid_roman_pairs).optimize()

        cardinal_graph = cardinal.graph

        graph = (
            pynutil.insert("key_cardinal: \"")
            + key_words_fst
            + pynutil.insert("\"")
            + pynini.accep(NEMO_SPACE)
            + pynutil.insert("integer: \"")
            + pynini.compose(roman_to_arabic_fst, cardinal_graph)
            + pynutil.insert("\"")
        ).optimize()

        self.fst = self.add_tokens(graph).optimize()

    def _int_to_roman(self, num):
        pass
