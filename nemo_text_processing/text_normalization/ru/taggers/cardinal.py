# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2017 Google Inc.
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

# Adapted from https://github.com/google/TextNormalizationCoveringGrammars
# Russian minimally supervised number grammar.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.ru.alphabet import RU_ALPHA, TO_CYRILLIC
from nemo_text_processing.text_normalization.ru.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        "1 001" ->  cardinal { integer: "Ñ‚Ñ‹Ñ�Ñ�Ñ‡Ð° Ð¾Ð´Ð¸Ð½" }

    Args:
        number_names: number_names for cardinal and ordinal numbers
        alternative_formats: alternative number formats
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, number_names: dict, alternative_formats: dict, deterministic: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        self.cardinal_numbers_default = self.get_cardinal_numbers(number_names, alternative_formats, mode="all")
        self.cardinal_numbers_nominative = self.get_cardinal_numbers(
            number_names, alternative_formats, mode="nominative"
        )
        self.optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space, 0, 1
        )

        self.cardinal_numbers_with_optional_negative = (
            self.optional_graph_negative
            + pynutil.insert("integer: \"")
            + self.cardinal_numbers_default
            + pynutil.insert("\"")
        )

        # "03" -> remove leading zeros and verbalize
        leading_zeros = pynini.closure(pynini.cross("0", ""))
        self.cardinal_numbers_with_leading_zeros = (leading_zeros + self.cardinal_numbers_default).optimize()

        # "123" -> "Ð¾Ð´Ð¸Ð½ Ð´Ð²Ð° Ñ‚Ñ€Ð¸"
        single_digits_graph = pynini.string_file(get_abs_path("data/numbers/cardinals_nominative_case.tsv")).optimize()
        single_digits_graph = pynini.compose(NEMO_DIGIT, single_digits_graph)
        self.single_digits_graph = single_digits_graph + pynini.closure(insert_space + single_digits_graph)

        optional_quantity = pynini.string_file(get_abs_path("data/numbers/quantity.tsv")).optimize()
        optional_quantity = pynutil.insert("quantity: \"") + optional_quantity + pynutil.insert("\"")
        optional_quantity = pynini.closure(
            (pynutil.add_weight(pynini.accep(NEMO_SPACE), -0.1) | insert_space) + optional_quantity, 0, 1
        )

        serial_graph = self.get_serial_graph()

        final_graph = (
            self.optional_graph_negative
            + pynutil.insert("integer: \"")
            + self.cardinal_numbers_with_leading_zeros
            + pynutil.insert("\"")
            + optional_quantity
        ).optimize()

        final_graph = pynutil.add_weight(final_graph, -0.1)
        final_graph |= pynutil.insert("integer: \"") + pynutil.add_weight(serial_graph, 10) + pynutil.insert("\"")
        self.final_graph = final_graph

        # to cover cases "2-Ñ…" -> "Ð´Ð²ÑƒÑ…" (this is not covered by ordinal endings)
        final_graph |= pynini.compose(
            pynini.compose(NEMO_DIGIT ** (1, ...) + pynini.cross('-Ñ…', ''), final_graph),
            NEMO_SIGMA + pynini.accep("Ñ…\"") + NEMO_SIGMA,
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

    def get_cardinal_numbers(self, number_names: dict, alternative_formats: dict, mode: str = "all"):
        """Returns cardinal numbers names graph.

        Args:
            number_names: number_names for cardinal and ordinal numbers
            alternative_formats: alternative number formats
            mode: "all" - to return graph that includes all Ru cases, "nominative" to return only the nominative form
        """
        pass

    def get_serial_graph(self):
        """
        Finite state transducer for classifying serial.
            The serial is a combination of digits, letters and dashes, e.g.:
            c325-b -> tokens { cardinal { integer: "Ñ�Ð¸ Ñ‚Ñ€Ð¸ Ð´Ð²Ð° Ð¿Ñ�Ñ‚ÑŒ Ð±Ð¸" } }
        """
        pass
