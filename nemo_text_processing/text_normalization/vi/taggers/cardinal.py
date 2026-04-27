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

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_DIGIT, GraphFst, insert_space
from nemo_text_processing.text_normalization.vi.utils import get_abs_path, load_labels


class CardinalFst(GraphFst):
    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        resources = {
            'zero': pynini.string_file(get_abs_path("data/numbers/zero.tsv")),
            'digit': pynini.string_file(get_abs_path("data/numbers/digit.tsv")),
            'teen': pynini.string_file(get_abs_path("data/numbers/teen.tsv")),
            'ties': pynini.string_file(get_abs_path("data/numbers/ties.tsv")),
        }
        self.zero, self.digit, self.teen, self.ties = resources.values()

        magnitudes_labels = load_labels(get_abs_path("data/numbers/magnitudes.tsv"))
        self.magnitudes = {parts[0]: parts[1] for parts in magnitudes_labels if len(parts) == 2}

        digit_special_labels = load_labels(get_abs_path("data/numbers/digit_special.tsv"))
        special = {parts[0]: {'std': parts[1], 'alt': parts[2]} for parts in digit_special_labels if len(parts) >= 3}

        self.special_digits = pynini.union(
            *[pynini.cross(k, v["alt"]) for k, v in special.items() if k in ["1", "4", "5"]]
        )
        self.linh_digits = pynini.union(*[pynini.cross(k, special[k]["std"]) for k in ["1", "4", "5"]], self.digit)

        self.two_digit = pynini.union(
            self.teen,
            self.ties + pynutil.delete("0"),
            self.ties
            + insert_space
            + pynini.union(self.special_digits, pynini.union("2", "3", "6", "7", "8", "9") @ self.digit),
        )

        hundred_word = self.magnitudes["hundred"]
        linh_word = self.magnitudes["linh"]

        # X00: má»™t trÄƒm, hai trÄƒm, etc.
        hundreds_exact = self.digit + insert_space + pynutil.insert(hundred_word) + pynutil.delete("00")

        # X0Y: má»™t trÄƒm linh má»™t, hai trÄƒm linh nÄƒm, etc.
        hundreds_with_linh = (
            self.digit
            + insert_space
            + pynutil.insert(hundred_word)
            + pynutil.delete("0")
            + insert_space
            + pynutil.insert(linh_word)
            + insert_space
            + self.linh_digits
        )

        # XYZ: má»™t trÄƒm hai mÆ°á»�i ba, etc.
        hundreds_with_tens = self.digit + insert_space + pynutil.insert(hundred_word) + insert_space + self.two_digit

        # 0YZ: Handle numbers starting with 0 (e.g., 087 -> tÃ¡m mÆ°Æ¡i báº£y)
        leading_zero_tens = pynutil.delete("0") + self.two_digit

        # 00Z: Handle numbers starting with 00 (e.g., 008 -> tÃ¡m)
        leading_double_zero = pynutil.delete("00") + self.digit

        self.hundreds_pattern = pynini.union(
            hundreds_exact,
            hundreds_with_linh,
            hundreds_with_tens,
            leading_zero_tens,
            leading_double_zero,
        )

        self.hundreds = pynini.closure(NEMO_DIGIT, 3, 3) @ self.hundreds_pattern

        self.magnitude_patterns = self._build_all_magnitude_patterns()
        custom_patterns = self._build_all_patterns()

        all_patterns = [
            *custom_patterns,
            *self.magnitude_patterns.values(),
            self.hundreds,
            self.two_digit,
            self.digit,
            self.zero,
        ]
        self.graph = pynini.union(*all_patterns).optimize()

        self.single_digits_graph = self.digit | self.zero

        negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        final_graph = negative + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        self.fst = self.add_tokens(final_graph).optimize()

    def _build_magnitude_pattern(self, name, min_digits, max_digits, zero_count, prev_pattern=None):
        pass

    def _build_all_magnitude_patterns(self):
        pass

    def _get_zero_or_magnitude_pattern(self, digits, magnitude_key):
        """Create pattern that handles all-zeros or normal magnitude processing"""
        pass

    def _build_all_patterns(self):
        pass

    def _get_pattern_for_digits(self, digit_count):
        pass
