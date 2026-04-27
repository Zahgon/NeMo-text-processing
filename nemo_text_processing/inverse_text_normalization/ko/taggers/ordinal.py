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


import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import GraphFst, delete_space
from nemo_text_processing.inverse_text_normalization.ko.utils import get_abs_path


def get_counter(ordinal):
    # counter suffix file (ê°œ, ëª…, ë³‘, ë§ˆë¦¬, ...)
    pass


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        Expressing integers in ordinal way for 1-39 and cardinal for 40+ due to Korean grammar.
        e.g. ìŠ¤ë¬¼ì„¸ë²ˆì§¸ -> ordinal {integer: "23", 23ë²ˆì§¸}
        e.g. ì‚¬ì‹­ì˜¤ë²ˆì§¸ -> ordinal but the integer part is written in cardinal(due to korean grammar)
        { integer: "45", 45ë²ˆì¨°}
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        cardinals = cardinal.just_cardinals
        man_as_10000 = pynini.cross("ë§Œ", "10000")
        ordinals_suffix = pynini.accep("ë²ˆì§¸")  # Korean ordinal's morphosyntactic feature

        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))  # 1-9 in ordinals
        cardinal_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))  # 1-9 in cardinals

        graph_tens_prefix = pynini.cross("ì—´", "1")  # First digit for tens
        graph_twenties_prefix = pynini.cross("ìŠ¤ë¬¼", "2")  # First digit for twenties
        graph_thirties_prefix = pynini.cross("ì„œë¥¸", "3")  # First digit for thirties

        # Below exclude regular 1 in ordinal and replace with a special 1. Like "first" in English
        # The special 1 is a unique ordinal case for Korean and does not repeat for 11, 21, 31
        graph_one = pynini.cross("í•œ", "1")
        single_digits = pynini.project(graph_digit, "input").optimize()
        graph_one_acceptor = pynini.project(graph_one, "input").optimize()
        two_to_nine = pynini.difference(single_digits, graph_one_acceptor).optimize()
        graph_two_to_nine = two_to_nine @ graph_digit
        graph_first = pynini.cross("ì²«", "1")
        graph_single = graph_two_to_nine | graph_first

        graph_ten = pynini.cross("ì—´", "10")
        graph_tens = graph_ten | graph_tens_prefix + graph_digit

        graph_twenty = pynini.cross("ìŠ¤ë¬´", "20")
        graph_twenties = graph_twenty | graph_twenties_prefix + graph_digit

        graph_thirty = pynini.cross("ì„œë¥¸", "30")
        graph_thirties = graph_thirty | graph_thirties_prefix + graph_digit

        ordinals = pynini.union(
            graph_single, graph_tens, graph_twenties, graph_thirties  # 1-9  # 10-19  # 20-29  # 30-39
        ).optimize()

        cardinal_10_to_19 = pynini.cross("ì‹­", "10") | (pynini.accep("ì‹­") + cardinal_digit)

        cardinal_20_to_29 = pynini.cross("ì�´ì‹­", "20") | (pynini.accep("ì�´ì‹­") + cardinal_digit)

        cardinal_30_to_39 = pynini.cross("ì‚¼ì‹­", "30") | (pynini.accep("ì‚¼ì‹­") + cardinal_digit)

        # FST that include 1-39 in cardinal expression
        cardinal_below_40 = pynini.union(
            cardinal_digit, cardinal_10_to_19, cardinal_20_to_29, cardinal_30_to_39
        ).optimize()

        # Input includes all cardinal expressions
        cardinals_acceptor = pynini.project(cardinals, "input").optimize()
        # Input includes cardinal expression from 1 to 39
        cardinals_exception = pynini.project(cardinal_below_40, "input").optimize()

        # All cardinal values except 1 to 39 cardinal values
        cardinal_over_40 = pynini.difference(cardinals_acceptor, cardinals_exception).optimize()
        cardinal_ordinal_suffix = cardinal_over_40 @ cardinals

        # 1 to 39 in ordinal, everything else cardinal
        ordinal_final = pynini.union(ordinals, cardinal_ordinal_suffix, man_as_10000)

        ordinal_graph = (
            pynutil.insert("integer: \"") + ((ordinal_final + delete_space + ordinals_suffix)) + pynutil.insert("\"")
        )

        # Adding various counter suffix for ordinal
        # For counting, Korean does not use the speical "ì²«" for 1. Instead the regular "í•œ"
        counters = pynini.union(graph_digit, graph_tens, graph_twenties, graph_thirties).optimize()

        counter_final = get_counter(counters) | get_counter(cardinal_ordinal_suffix) | get_counter(man_as_10000)

        counter_graph = pynutil.insert("integer: \"") + counter_final + pynutil.insert("\"")

        final_graph = ordinal_graph | counter_graph

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
