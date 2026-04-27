# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    NEMO_CHAR,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path

shunya = (
    pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
    | pynini.string_file(get_abs_path("data/telephone/eng_zero.tsv")).invert()
)
digit_without_shunya = (
    pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
    | pynini.string_file(get_abs_path("data/telephone/eng_digit.tsv")).invert()
)
digit = digit_without_shunya | shunya


def get_context(keywords: list):
    pass


def generate_context_graph(context_keywords, length):
    pass


def generate_pincode(context_keywords):
    pass


def generate_credit(context_keywords):
    pass


def generate_mobile(context_keywords):
    pass


def generate_telephone(context_keywords):
    pass


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g.
    e.g. à¤ªà¥�à¤²à¤¸ à¤‡à¤•à¥�à¤¯à¤¾à¤¨à¤µà¥‡ à¤¨à¥Œ à¤†à¤  à¤¸à¤¾à¤¤ à¤›à¤¹ à¤ªà¤¾à¤‚à¤š à¤šà¤¾à¤° à¤¤à¥€à¤¨ à¤¦à¥‹ à¤�à¤• à¤¶à¥‚à¤¨à¥�à¤¯ => tokens { name: "+à¥¯à¥§ à¥¯à¥®à¥­à¥¬à¥« à¥ªà¥©à¥¨à¥§à¥¦" }
    Args:
        Cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="telephone", kind="classify")

        # Load context cues from TSV file
        context_cues = pynini.string_file(get_abs_path("data/telephone/context_cues.tsv"))

        # Extract keywords for each category
        mobile_keywords = pynini.compose(pynutil.delete("mobile"), context_cues).project("output").optimize()

        landline_keywords = pynini.compose(pynutil.delete("landline"), context_cues).project("output").optimize()

        pincode_keywords = pynini.compose(pynutil.delete("pincode"), context_cues).project("output").optimize()

        credit_keywords = pynini.compose(pynutil.delete("credit"), context_cues).project("output").optimize()

        # Convert FSTs to keyword lists for generate_* functions
        mobile = generate_mobile([mobile_keywords])
        landline = generate_telephone([landline_keywords])
        pincode = generate_pincode([pincode_keywords])
        credit = generate_credit([credit_keywords])

        graph = (
            pynutil.add_weight(mobile, 0.7)
            | pynutil.add_weight(landline, 0.8)
            | pynutil.add_weight(credit, 0.9)
            | pynutil.add_weight(pincode, 1)
        )

        self.final = graph.optimize()
        self.fst = self.add_tokens(self.final)
