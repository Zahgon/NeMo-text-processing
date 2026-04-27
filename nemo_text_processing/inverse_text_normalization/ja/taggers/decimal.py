# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ja.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.ja.utils import get_abs_path


def get_quantity(decimal):
    pass


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal
        e.g. ä¸€ç‚¹äº” -> decimnl { integer_part: "1" fractional_part: "5" }
        e.g. ä¸€ç‚¹äº”ä¸‡ -> decimal { integer_part: "1" fractional_part: "5" quantity: "ä¸‡" }
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        cardinals = cardinal.just_cardinals
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        after_decimal = pynini.closure(graph_zero | graph_digit)

        decimal_point = pynutil.delete("ç‚¹")
        fractional_component = pynutil.insert("fractional_part: \"") + after_decimal + pynutil.insert("\"")
        integer_component = pynutil.insert("integer_part: \"") + cardinals + pynutil.insert("\"")

        graph_decimal_regular = integer_component + decimal_point + pynutil.insert(" ") + fractional_component
        graph_deicimal_larger = get_quantity(graph_decimal_regular)

        self.decimal = graph_decimal_regular | graph_deicimal_larger
        self.just_decimal = cardinals + pynini.cross("ç‚¹", ".") + after_decimal

        graph_sign = (
            pynutil.insert("negative: \"") + (pynini.cross("ãƒžã‚¤ãƒŠã‚¹", "-") | pynini.accep("-")) + pynutil.insert("\"")
        )

        final_graph = (
            (graph_sign + pynutil.insert(" ") + graph_decimal_regular)
            | (graph_sign + pynutil.insert(" ") + graph_deicimal_larger)
            | graph_decimal_regular
            | graph_deicimal_larger
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
