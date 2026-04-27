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


import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst
from nemo_text_processing.text_normalization.zh.utils import get_abs_path


def get_quantity(decimal):
    pass


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        0.5 -> decimal { integer_part: "é›¶" fractional_part: "äº”" }
        0.5ä¸‡ -> decimal { integer_part: "é›¶" fractional_part: "äº”" quantity: "ä¸‡" }
        -0.5ä¸‡ -> decimal { negative: "è´Ÿ" integer_part: "é›¶" fractional_part: "äº”" quantity: "ä¸‡"}

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True, lm: bool = False):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        cardinal_before_decimal = cardinal.just_cardinals
        cardinal_after_decimal = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))

        graph_integer = pynutil.insert('integer_part: \"') + cardinal_before_decimal + pynutil.insert("\"")

        graph_fraction = (
            pynutil.insert("fractional_part: \"")
            + pynini.closure((pynini.closure(cardinal_after_decimal, 1) | (pynini.closure(zero, 1))), 1)
            + pynutil.insert("\"")
        )
        graph_decimal = graph_integer + pynutil.delete('.') + pynutil.insert(" ") + graph_fraction
        self.regular_decimal = graph_decimal.optimize()

        graph_sign = (
            (
                pynini.closure(pynutil.insert("negative: \"") + pynini.cross("-", "è´Ÿ"))
                + pynutil.insert("\"")
                + pynutil.insert(" ")
            )
        ) | (
            (
                pynutil.insert('negative: ')
                + pynutil.insert("\"")
                + (pynini.accep('è´Ÿ') | pynini.cross('è² ', 'è´Ÿ'))
                + pynutil.insert("\"")
                + pynutil.insert(' ')
            )
        )
        graph_with_sign = graph_sign + graph_decimal
        graph_regular = graph_with_sign | graph_decimal

        graph_decimal_quantity = get_quantity(graph_decimal)
        graph_sign_quantity = graph_sign + graph_decimal_quantity
        graph_quantity = graph_decimal_quantity | graph_sign_quantity

        final_graph = graph_regular | graph_quantity
        self.decimal = final_graph.optimize()

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
