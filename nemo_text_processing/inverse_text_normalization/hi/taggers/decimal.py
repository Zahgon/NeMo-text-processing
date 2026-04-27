# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
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
# limitations under the License.import pynini

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MIN_NEG_WEIGHT,
    MINUS,
    NEMO_DIGIT,
    NEMO_SIGMA,
    TO_LOWER,
    GraphFst,
    capitalized_input_graph,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.text_normalization.en.utils import load_labels


def get_quantity(
    decimal: 'pynini.FstLike', cardinal_up_to_hundred: 'pynini.FstLike', input_case: str = INPUT_LOWER_CASED
) -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. à¤¦à¤¸ à¤²à¤¾à¤– -> integer_part: "à¥§à¥°" quantity: "à¤²à¤¾à¤–"
    e.g. à¤�à¤• à¤¦à¤¶à¤®à¤²à¤µ à¤ªà¤¾à¤�à¤š à¤²à¤¾à¤– -> integer_part: "à¥§" fractional_part: "à¥«" quantity: "à¤²à¤¾à¤–"

    Args:
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
        input_case: accepting either "lower_cased" or "cased" input.
    """
    pass


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal
        Decimal point "." is determined by "à¤¦à¤¶à¤®à¤²à¤µ"
            e.g. à¤‹à¤£ à¤�à¤• à¤¦à¤¶à¤®à¤²à¤µ à¤¦à¥‹ à¤›à¤¹ -> decimal { negative: "true" integer_part: "à¥§" morphosyntactic_features: "." fractional_part: "à¥¨à¥¬" }


        This decimal rule assumes that decimals can be pronounced as:
        (a cardinal) + ('à¤¦à¤¶à¤®à¤²à¤µ') plus (any sequence of cardinals <à¥§à¥¦à¥¦à¥¦, including 'à¤¶à¥‚à¤¨à¥�à¤¯')

        Also writes large numbers in shortened form, e.g.
            e.g. à¤�à¤• à¤¦à¤¶à¤®à¤²à¤µ à¤¦à¥‹ à¤›à¤¹ à¤²à¤¾à¤– -> decimal { negative: "false" integer_part: "à¥§" morphosyntactic_features: "." fractional_part: "à¥¨à¥¬" quantity: "à¤²à¤¾à¤–" }
            e.g. à¤¦à¥‹ à¤²à¤¾à¤– -> decimal { negative: "false" integer_part: "à¥¨" quantity: "à¤²à¤¾à¤–" }
            e.g. à¤�à¤• à¤…à¤°à¤¬ à¤†à¤  à¤¸à¥Œ à¤šà¥Œà¤¬à¥€à¤¸ à¤²à¤¾à¤– -> decimal { negative: "false" integer_part: "à¥§à¥®à¥¨à¥ª" quantity: "à¤²à¤¾à¤–" }
    Args:
        cardinal: CardinalFst

    """

    def __init__(self, cardinal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="decimal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        graph_decimal = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_decimal |= pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()

        graph_decimal = pynini.closure(graph_decimal + delete_space) + graph_decimal
        self.graph = graph_decimal

        point = pynutil.delete("à¤¦à¤¶à¤®à¤²à¤µ")

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("à¤‹à¤£", "\"true\"") + delete_extra_space,
            0,
            1,
        )

        graph_fractional = pynutil.insert("fractional_part: \"") + graph_decimal + pynutil.insert("\"")
        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        final_graph_wo_sign = (
            pynini.closure(graph_integer + delete_extra_space, 0, 1) + point + delete_extra_space + graph_fractional
        )
        final_graph = optional_graph_negative + final_graph_wo_sign

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(
            final_graph_wo_sign, cardinal_graph, input_case=input_case
        )

        # accept semiotic spans that start with a capital letter
        self.final_graph_wo_negative |= pynutil.add_weight(
            pynini.compose(TO_LOWER + NEMO_SIGMA, self.final_graph_wo_negative).optimize(), MIN_NEG_WEIGHT
        )

        quantity_graph = get_quantity(final_graph_wo_sign, cardinal_graph, input_case=input_case)
        final_graph |= optional_graph_negative + quantity_graph

        if input_case == INPUT_CASED:
            final_graph = capitalized_input_graph(final_graph)
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
