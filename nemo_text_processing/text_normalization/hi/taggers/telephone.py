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

from nemo_text_processing.text_normalization.hi.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

HI_ZERO_DIGIT = pynini.union("0", "à¥¦")
HI_MOBILE_START_DIGITS = pynini.union("à¥¬", "à¥­", "à¥®", "à¥¯", "6", "7", "8", "9").optimize()
HI_LANDLINE_START_DIGITS = pynini.union("à¥¨", "à¥©", "à¥ª", "à¥¬", "2", "3", "4", "6").optimize()

delete_zero = pynutil.delete(HI_ZERO_DIGIT)
delete_zero_optional = pynini.closure(delete_zero, 0, 1)
insert_shunya = pynutil.insert('à¤¶à¥‚à¤¨à¥�à¤¯') + insert_space

# Load the number mappings from the TSV file
digit_to_word = pynini.string_file(get_abs_path("data/telephone/number.tsv"))
digits = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
mobile_context = pynini.string_file(get_abs_path("data/telephone/mobile_context.tsv"))
landline_context = pynini.string_file(get_abs_path("data/telephone/landline_context.tsv"))
credit_context = pynini.string_file(get_abs_path("data/telephone/credit_context.tsv"))
pincode_context = pynini.string_file(get_abs_path("data/telephone/pincode_context.tsv"))

# Reusable optimized graph for any digit token
num_token = pynini.union(digit_to_word, digits, zero).optimize()


def generate_mobile(context_keywords: pynini.Fst) -> pynini.Fst:
    pass


def get_landline(std_length: int, context_keywords: pynini.Fst) -> pynini.Fst:
    pass


def generate_landline(context_keywords: pynini.Fst) -> pynini.Fst:
    pass


def get_context(keywords: pynini.Fst):

    pass


def generate_credit(context_keywords: pynini.Fst) -> pynini.Fst:
    pass


def generate_pincode(context_keywords: pynini.Fst) -> pynini.Fst:
    pass


class TelephoneFst(GraphFst):
    """
    Finite state transducer for tagging telephone numbers, e.g.
        à¥¯à¥§à¥«à¥­à¥§à¥§à¥ªà¥¦à¥¦à¥­ -> telephone { number_part: "à¤¶à¥‚à¤¨à¥�à¤¯ à¤¨à¥Œ à¤�à¤• à¤ªà¤¾à¤�à¤š à¤¸à¤¾à¤¤ à¤�à¤• à¤�à¤• à¤šà¤¾à¤° à¤¶à¥‚à¤¨à¥�à¤¯ à¤¶à¥‚à¤¨à¥�à¤¯ à¤¸à¤¾à¤¤" }
        +à¥¯à¥§ à¥¯à¥¨à¥§à¥¦à¥«à¥§à¥«à¥¬à¥¦à¥¬ -> telephone { country_code: "à¤ªà¥�à¤²à¤¸ à¤¨à¥Œ à¤�à¤•", number_part: "à¤¨à¥Œ à¤¦à¥‹ à¤�à¤• à¤¶à¥‚à¤¨à¥�à¤¯ à¤ªà¤¾à¤�à¤š à¤�à¤• à¤ªà¤¾à¤�à¤š à¤›à¤¹ à¤¶à¥‚à¤¨à¥�à¤¯ à¤›à¤¹" }
        à¥§à¥©à¥­à¥ª-à¥©à¥¦à¥¯à¥¯à¥®à¥® -> telephone { number_part: "à¤¶à¥‚à¤¨à¥�à¤¯ à¤�à¤• à¤¤à¥€à¤¨ à¤¸à¤¾à¤¤ à¤šà¤¾à¤° à¤¤à¥€à¤¨ à¤¶à¥‚à¤¨à¥�à¤¯ à¤¨à¥Œ à¤¨à¥Œ à¤†à¤  à¤†à¤ " }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization
    """

    def __init__(self):
        super().__init__(name="telephone", kind="classify")

        mobile_number = generate_mobile(mobile_context)
        landline = generate_landline(landline_context)
        credit_card = generate_credit(credit_context)
        pincode = generate_pincode(pincode_context)

        graph = (
            pynutil.add_weight(mobile_number, 0.7)
            | pynutil.add_weight(landline, 0.8)
            | pynutil.add_weight(credit_card, 0.9)
            | pynutil.add_weight(pincode, 1)
        )

        self.final = graph.optimize()
        self.fst = self.add_tokens(self.final)
