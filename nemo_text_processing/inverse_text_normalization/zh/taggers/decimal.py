# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.zh.utils import get_abs_path


def get_quantity(decimal, cardinal):
    pass


class DecimalFst(GraphFst):
    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        cardinal_after_decimal = pynini.string_file(get_abs_path("data/numbers/digit-nano.tsv")) | pynini.closure(
            pynini.cross("é›¶", "0")
        )
        cardinal_before_decimal = cardinal.just_cardinals | pynini.cross("é›¶", "0")

        delete_decimal = pynutil.delete("ç‚¹") | pynutil.delete("é»ž")

        graph_integer = pynutil.insert('integer_part: "') + cardinal_before_decimal + pynutil.insert('" ')

        graph_string_of_cardinals = pynini.closure(cardinal_after_decimal, 1)
        graph_fractional = pynutil.insert('fractional_part: "') + graph_string_of_cardinals + pynutil.insert('"')

        graph_decimal_no_sign = pynini.closure((graph_integer + delete_decimal + graph_fractional), 1)

        self.final_graph_wo_negative = graph_decimal_no_sign | get_quantity(
            graph_decimal_no_sign, cardinal.just_cardinals
        )

        graph_negative = pynini.cross("è´Ÿ", 'negative: "-" ') | pynini.cross("è² ", 'negative: "-" ')
        graph_negative = pynini.closure(graph_negative, 0, 1)  # captures only one "è´Ÿ"

        graph_decimal = graph_negative + graph_decimal_no_sign
        graph_decimal = graph_decimal | (graph_negative + get_quantity(graph_decimal_no_sign, cardinal_before_decimal))
        self.final_graph_wo_negative = graph_decimal

        final_graph = self.add_tokens(graph_decimal)
        self.fst = final_graph.optimize()
