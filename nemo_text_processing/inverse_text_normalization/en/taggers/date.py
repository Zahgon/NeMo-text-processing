# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.en.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    NEMO_ALPHA,
    NEMO_DIGIT,
    GraphFst,
    capitalized_input_graph,
    delete_extra_space,
    delete_space,
)

graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv")).optimize()
graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize()
ties_graph = pynini.string_file(get_abs_path("data/numbers/ties.tsv")).optimize()


def _get_month_graph(input_case: str = INPUT_LOWER_CASED):
    """
    Transducer for month, e.g. march -> march
    """
    pass


def _get_ties_graph(input_case: str):
    """
    Transducer for 20-99 e.g
    twenty three -> 23
    """
    pass


def _get_range_graph(input_case: str):
    """
    Transducer for decades (1**0s, 2**0s), centuries (2*00s, 1*00s), millennia (2000s)
    """
    pass


def _get_year_graph(input_case: str):
    """
    Transducer for year, e.g. twenty twenty -> 2020
    """
    pass


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date,
        e.g. january fifth twenty twelve -> date { month: "january" day: "5" year: "2012" preserve_order: true }
        e.g. the fifth of january twenty twelve -> date { day: "5" month: "january" year: "2012" preserve_order: true }
        e.g. twenty twenty -> date { year: "2012" preserve_order: true }

    Args:
        ordinal: OrdinalFst
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, ordinal: GraphFst, input_case: str):
        super().__init__(name="date", kind="classify")

        ordinal_graph = ordinal.graph
        year_graph = _get_year_graph(input_case=input_case)
        YEAR_WEIGHT = 0.001
        year_graph = pynutil.add_weight(year_graph, YEAR_WEIGHT)
        month_graph = _get_month_graph(input_case=input_case)

        month_graph = pynutil.insert("month: \"") + month_graph + pynutil.insert("\"")

        day_graph = pynutil.insert("day: \"") + pynutil.add_weight(ordinal_graph, -0.7) + pynutil.insert("\"")
        graph_year = (
            delete_extra_space
            + pynutil.insert("year: \"")
            + pynutil.add_weight(year_graph, -YEAR_WEIGHT)
            + pynutil.insert("\"")
        )
        optional_graph_year = pynini.closure(
            graph_year,
            0,
            1,
        )
        graph_mdy = month_graph + (
            (delete_extra_space + day_graph) | graph_year | (delete_extra_space + day_graph + graph_year)
        )
        the_graph = pynutil.delete("the")
        if input_case == INPUT_CASED:
            the_graph |= pynutil.delete("The").optimize()

        graph_dmy = (
            the_graph
            + delete_space
            + day_graph
            + delete_space
            + pynutil.delete("of")
            + delete_extra_space
            + month_graph
            + optional_graph_year
        )

        financial_period_graph = pynini.string_file(get_abs_path("data/date_period.tsv")).invert()
        period_fy = (
            pynutil.insert("text: \"")
            + financial_period_graph
            + (pynini.cross(" ", "") | pynini.cross(" of ", ""))
            + pynutil.insert("\"")
        )

        graph_year = (
            pynutil.insert("year: \"") + (year_graph | _get_range_graph(input_case=input_case)) + pynutil.insert("\"")
        )

        graph_fy = period_fy + pynutil.insert(" ") + graph_year

        final_graph = graph_mdy | graph_dmy | graph_year | graph_fy
        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
