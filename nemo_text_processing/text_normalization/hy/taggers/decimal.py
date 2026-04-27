# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.hy.utils import get_abs_path


def get_quantity(decimal_graph: "pynini.FstLike", cardinal_graph: "pynini.FstLike") -> "pynini.FstLike":
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. 2 Õ´Õ«Õ¬Õ«Õ¸Õ¶ -> integer_part: "Õ¥Ö€Õ¯Õ¸Ö‚" quantity: "Õ´Õ«Õ¬Õ«Õ¸Õ¶"
    e.g. 2â€¤4 Õ´Õ«Õ¬Õ«Õ¸Õ¶ -> integer_part: "Õ¥Ö€Õ¯Õ¸Ö‚" fractional_part: "Õ¹Õ¸Ö€Õ½" quantity: "Õ´Õ«Õ¬Õ«Õ¸Õ¶"
    Args:
        decimal_graph: DecimalFST
        cardinal_graph: CardinalFST
    """
    pass


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        554 Õ´Õ«Õ¬Õ«Õ¡Ö€Õ¤ -> decimal { integer_part: "Õ°Õ«Õ¶Õ£ Õ°Õ¡Ö€ÕµÕ¸Ö‚Ö€ Õ°Õ«Õ½Õ¸Ö‚Õ¶Õ¹Õ¸Ö€Õ½" quantity: "Õ´Õ«Õ¬Õ«Õ¡Ö€Õ¤" }
    Args:
        cardinal: CardinalFst
        deterministic is not necessary right now
        TODO make deterministic make sense
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        graph = cardinal.one_to_all_tens

        graph = graph.optimize()

        delete_separator = pynutil.delete(".") | pynutil.delete("â€¤")
        optional_graph_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1)

        graph_fractional = pynutil.insert('fractional_part: "') + graph + pynutil.insert('"')

        integers = cardinal.all_nums_no_tokens
        graph_integer = pynutil.insert('integer_part: "') + integers + pynutil.insert('"')
        final_graph_wo_sign = graph_integer + delete_separator + insert_space + graph_fractional

        final_graph_wo_negative = final_graph_wo_sign | get_quantity(final_graph_wo_sign, integers)
        self.final_graph_wo_negative = final_graph_wo_negative.optimize()

        final_graph = optional_graph_negative + final_graph_wo_negative

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
