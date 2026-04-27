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

import os
from typing import Dict, List

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_SIGMA, NEMO_SPACE, generator_main
from nemo_text_processing.utils.logging import logger


class PostProcessingFst:
    """
    Finite state transducer that post-processes an entire Vietnamese sentence after verbalization is complete, e.g.
    removes extra spaces around punctuation marks " ( má»™t trÄƒm hai mÆ°Æ¡i ba ) " -> "(má»™t trÄƒm hai mÆ°Æ¡i ba)"

    Args:
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
    """

    def __init__(self, cache_dir: str = None, overwrite_cache: bool = False):

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, "vi_tn_post_processing.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["post_process_graph"]
            logger.info(f'Post processing graph was restored from {far_file}.')
        else:
            self.set_punct_dict()
            self.fst = self.get_punct_postprocess_graph()

            if far_file:
                generator_main(far_file, {"post_process_graph": self.fst})

    def get_vietnamese_punct_config(self) -> Dict[str, List[str]]:
        """
        Returns Vietnamese-specific punctuation configuration.
        This method can be easily modified or extended for different Vietnamese punctuation rules.
        """
        pass

    def set_punct_dict(self):
        # Vietnamese punctuation marks that might need special handling
        pass

    def get_punct_postprocess_graph(self):
        """
        Returns graph to post process punctuation marks for Vietnamese.

        Uses dynamic configuration for flexible punctuation handling.
        Vietnamese punctuation spacing rules are defined in get_vietnamese_punct_config().
        """
        pass
