# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2017 Google Inc.
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

# Adapted from https://github.com/google/TextNormalizationCoveringGrammars
# Russian minimally supervised number grammar.
#
# Supports cardinals and ordinals in all inflected forms.
#
# The language-specific acceptor G was compiled with digit, teen, decade,
# century, and big power-of-ten preterminals. The lexicon transducer is
# highly ambiguous, but no LM is used.

# Intersects the universal factorization transducer (F) with language-specific
# acceptor (G).

import pynini
from pynini.lib import pynutil, rewrite

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA
from nemo_text_processing.text_normalization.ru.utils import get_abs_path, load_labels


def get_number_names():
    """
    Creates numbers names.

    Based on: 1) Gorman, K., and Sproat, R. 2016. Minimally supervised number normalization.
    Transactions of the Association for Computational Linguistics 4: 507-519.
    and 2) Ng, A. H., Gorman, K., and Sproat, R. 2017.
    Minimally supervised written-to-spoken text normalization. In ASRU, pages 665-670.
    """
    pass


def get_alternative_formats():
    """
    Utils to get alternative formats for numbers.
    """
    pass


if __name__ == '__main__':
    from nemo_text_processing.text_normalization.en.graph_utils import generator_main

    numbers = get_number_names()
    for k, v in numbers.items():
        generator_main(f'{k}.far', {k: v})
