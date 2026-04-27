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

from nemo_text_processing.text_normalization.hi.graph_utils import (
    ASTERISK,
    COMMA,
    DECIMAL_25,
    DECIMAL_75,
    HI_BY,
    HI_DEDH,
    HI_DHAI,
    HI_PAUNE,
    HI_PERIOD,
    HI_SADHE,
    HI_SAVVA,
    HYPHEN,
    INPUT_LOWER_CASED,
    LOWERCASE_X,
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    ONE_POINT_FIVE,
    PERIOD,
    POINT_FIVE,
    SLASH,
    TWO_POINT_FIVE,
    UPPERCASE_X,
    GraphFst,
    capitalized_input_graph,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
# Load both Hindi (Devanagari) and English (Arabic) number mappings
teens_ties_hi = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
teens_ties_en = pynini.string_file(get_abs_path("data/numbers/teens_and_ties_en.tsv"))
teens_ties = pynini.union(teens_ties_hi, teens_ties_en)
teens_and_ties = pynutil.add_weight(teens_ties, -0.1)


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, suppletive aware, e.g.
        -à¥§à¥¨kg -> measure { negative: "true" cardinal { integer: "à¤¬à¤¾à¤°à¤¹" } units: "à¤•à¤¿à¤²à¥‹à¤—à¥�à¤°à¤¾à¤®" }
        -à¥§à¥¨.à¥¨kg -> measure { decimal { negative: "true"  integer_part: "à¤¬à¤¾à¤°à¤¹"  fractional_part: "à¤¦à¥‹"} units: "à¤•à¤¿à¤²à¥‹à¤—à¥�à¤°à¤¾à¤®" }
        à¤®à¥�à¤‚à¤¬à¤ˆ à¥®à¥®à¥ªà¥ªà¥¦à¥ª -> measure { units: "address" cardinal { integer: "à¤®à¥�à¤‚à¤¬à¤ˆ à¤†à¤  à¤†à¤  à¤šà¤¾à¤° à¤šà¤¾à¤° à¤¶à¥‚à¤¨à¥�à¤¯ à¤šà¤¾à¤°" } preserve_order: true }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def get_structured_address_graph(self, ordinal: GraphFst, input_case: str):
        """
        Minimal address tagger for state/city + pincode patterns only.
        Highly optimized for performance.

        Examples:
            "à¤®à¥�à¤‚à¤¬à¤ˆ à¥®à¥®à¥ªà¥ªà¥¦à¥ª" -> "à¤®à¥�à¤‚à¤¬à¤ˆ à¤†à¤  à¤†à¤  à¤šà¤¾à¤° à¤šà¤¾à¤° à¤¶à¥‚à¤¨à¥�à¤¯ à¤šà¤¾à¤°"
            "à¤—à¥‹à¤µà¤¾ à¥§à¥¨à¥©à¥ªà¥«à¥¬" -> "à¤—à¥‹à¤µà¤¾ à¤�à¤• à¤¦à¥‹ à¤¤à¥€à¤¨ à¤šà¤¾à¤° à¤ªà¤¾à¤�à¤š à¤›à¤¹"
        """
        pass

    def get_address_graph(self, ordinal: GraphFst, input_case: str):
        """
        Address tagger that converts digits/hyphens/slashes character-by-character
        when address context keywords are present.
        English words and ordinals are converted to Hindi transliterations.

        Examples:
            "à¥­à¥¦à¥¦ à¤“à¤• à¤¸à¥�à¤Ÿà¥�à¤°à¥€à¤Ÿ" -> "à¤¸à¤¾à¤¤ à¤¶à¥‚à¤¨à¥�à¤¯ à¤¶à¥‚à¤¨à¥�à¤¯ à¤“à¤• à¤¸à¥�à¤Ÿà¥�à¤°à¥€à¤Ÿ"
            "à¥¬à¥¬-à¥ª à¤ªà¤¾à¤°à¥�à¤• à¤°à¥‹à¤¡" -> "à¤›à¤¹ à¤›à¤¹ à¤¹à¤¾à¤‡à¤«à¤¼à¤¨ à¤šà¤¾à¤° à¤ªà¤¾à¤°à¥�à¤• à¤°à¥‹à¤¡"
        """
        pass

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, ordinal: GraphFst, input_case: str):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = (
            cardinal.zero
            | cardinal.digit
            | cardinal.teens_and_ties
            | cardinal.graph_hundreds
            | cardinal.graph_thousands
            | cardinal.graph_ten_thousands
            | cardinal.graph_lakhs
            | cardinal.graph_ten_lakhs
        )
        point = pynutil.delete(".")
        decimal_integers = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        decimal_graph = decimal_integers + point + insert_space + decimal.graph_fractional
        unit_graph = pynini.string_file(get_abs_path("data/measure/unit.tsv"))

        # Year unit variants for formal/informal handling
        year_informal = pynini.string_map([("yr", "à¤¸à¤¾à¤²")])
        year_formal = pynini.string_file(get_abs_path("data/measure/unit_year_formal.tsv"))

        # All units EXCEPT year
        unit_inputs_except_yr = pynini.difference(pynini.project(unit_graph, "input"), pynini.accep("yr"))
        unit_graph_no_year = pynini.compose(unit_inputs_except_yr, unit_graph)

        # Load quarterly units from separate files: map (FST) and list (FSA)
        quarterly_units_map = pynini.string_file(get_abs_path("data/measure/quarterly_units_map.tsv"))
        quarterly_units_list = pynini.string_file(get_abs_path("data/measure/quarterly_units_list.tsv"))
        quarterly_units_graph = pynini.union(quarterly_units_map, quarterly_units_list)

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space,
            0,
            1,
        )

        # Define the quarterly measurements - support both Devanagari and Arabic digits
        quarter = pynini.union(
            pynini.cross(POINT_FIVE, HI_SADHE),
            pynini.cross(ONE_POINT_FIVE, HI_DEDH),
            pynini.cross(TWO_POINT_FIVE, HI_DHAI),
        )
        quarter_graph = pynutil.insert("integer_part: \"") + quarter + pynutil.insert("\"")

        # Define the unit handling
        unit = (
            pynutil.insert(NEMO_SPACE)
            + pynutil.insert("units: \"")
            + unit_graph_no_year
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )
        units = (
            pynutil.insert(NEMO_SPACE)
            + pynutil.insert("units: \"")
            + quarterly_units_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        # Year-specific unit wrappers
        unit_year_informal = (
            pynutil.insert(NEMO_SPACE)
            + pynutil.insert("units: \"")
            + year_informal
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )
        unit_year_formal = (
            pynutil.insert(NEMO_SPACE)
            + pynutil.insert("units: \"")
            + year_formal
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        # Cardinal >= 1000 -> formal year (à¤µà¤°à¥�à¤·)
        # Use graph_without_leading_zeros which covers all number ranges (thousands to shankhs)
        cardinal_large = cardinal.graph_without_leading_zeros

        # Cardinal < 1000 -> informal year (à¤¸à¤¾à¤²)
        cardinal_small = cardinal.zero | cardinal.digit | cardinal.teens_and_ties | cardinal.graph_hundreds

        symbol_graph = pynini.string_map(
            [
                (LOWERCASE_X, HI_BY),
                (UPPERCASE_X, HI_BY),
                (ASTERISK, HI_BY),
            ]
        )

        graph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_space
            + unit
        )

        # Support both Devanagari and Arabic digits for dedh/dhai patterns
        dedh_dhai = pynini.union(
            pynini.cross(ONE_POINT_FIVE, HI_DEDH),
            pynini.cross(TWO_POINT_FIVE, HI_DHAI),
        )
        dedh_dhai_graph = pynutil.insert("integer: \"") + dedh_dhai + pynutil.insert("\"")

        # Support both Devanagari and Arabic digits for savva pattern
        savva_numbers = cardinal_graph + pynini.cross(DECIMAL_25, "")
        savva_graph = (
            pynutil.insert("integer: \"")
            + pynutil.insert(HI_SAVVA)
            + pynutil.insert(NEMO_SPACE)
            + savva_numbers
            + pynutil.insert("\"")
        )

        # Support both Devanagari and Arabic digits for sadhe pattern
        sadhe_numbers = cardinal_graph + pynini.cross(POINT_FIVE, "")
        sadhe_graph = (
            pynutil.insert("integer: \"")
            + pynutil.insert(HI_SADHE)
            + pynutil.insert(NEMO_SPACE)
            + sadhe_numbers
            + pynutil.insert("\"")
        )

        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        # Support both Devanagari and Arabic digits for paune pattern
        paune_numbers = paune + pynini.cross(DECIMAL_75, "")
        paune_graph = (
            pynutil.insert("integer: \"")
            + pynutil.insert(HI_PAUNE)
            + pynutil.insert(NEMO_SPACE)
            + paune_numbers
            + pynutil.insert("\"")
        )

        graph_dedh_dhai = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + dedh_dhai_graph
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + units
        )

        graph_savva = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + savva_graph
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + units
        )

        graph_sadhe = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + sadhe_graph
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + units
        )

        graph_paune = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + paune_graph
            + pynutil.insert(" }")
            + delete_space
            + units
        )

        graph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + unit
        )

        # Large numbers (>=1000) + yr -> formal (à¤µà¤°à¥�à¤·)
        graph_cardinal_year_formal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_large
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + unit_year_formal
        )

        # Small numbers (<1000) + yr -> informal (à¤¸à¤¾à¤²)
        graph_cardinal_year_informal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_small
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + unit_year_informal
        )

        # Regular decimals (e.g., 16.07) + yr -> formal (à¤µà¤°à¥�à¤·)
        graph_decimal_year_formal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_space
            + unit_year_formal
        )

        # Handling cardinal clubbed with symbol as single token
        graph_exceptions = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("units: \"")
            + symbol_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("} }")
            + insert_space
            + pynutil.insert("tokens { cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
        )

        address_graph = self.get_address_graph(ordinal, input_case)
        structured_address_graph = self.get_structured_address_graph(ordinal, input_case)

        graph = (
            pynutil.add_weight(graph_decimal, 0.1)
            | pynutil.add_weight(graph_decimal_year_formal, 0.1)
            | pynutil.add_weight(graph_cardinal, 0.1)
            | pynutil.add_weight(graph_cardinal_year_formal, 0.1)
            | pynutil.add_weight(graph_cardinal_year_informal, -0.1)  # Higher priority for small numbers
            | pynutil.add_weight(graph_exceptions, 0.1)
            | pynutil.add_weight(graph_dedh_dhai, -0.2)
            | pynutil.add_weight(graph_savva, -0.1)
            | pynutil.add_weight(graph_sadhe, -0.1)
            | pynutil.add_weight(graph_paune, -0.5)
            | address_graph
            | structured_address_graph
        )
        self.graph = graph.optimize()

        final_graph = self.add_tokens(graph)
        self.fst = final_graph
