# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from typing import Dict

from cdifflib import CSequenceMatcher

from nemo_text_processing.utils.logging import logger

MATCH = "match"
NONMATCH = "non-match"
SEMIOTIC_TAG = "[SEMIOTIC_SPAN]"


def _get_alignment(a: str, b: str) -> Dict:
    """
    Constructs alignment between a and b

    Returns:
        a dictionary, where keys are a's word index and values is a Tuple that contains span from b, and whether it
            matches a or not, e.g.:
                >>> a = "a b c"
                >>> b = "a b d f"
                >>> print(_get_alignment(a, b))
                {0: (0, 1, 'match'), 1: (1, 2, 'match'), 2: (2, 4, 'non-match')}
    """
    pass


def adjust_boundaries(norm_raw_diffs: Dict, norm_pred_diffs: Dict, raw: str, norm: str, pred_text: str, verbose=False):
    """
    Adjust alignment boundaries by taking norm--raw texts and norm--pred_text alignments, and creating raw-pred_text alignment
        alignment.

    norm_raw_diffs: output of _get_alignment(norm, raw)
    norm_pred_diffs: output of  _get_alignment(norm, pred_text)
    raw: input text
    norm: output of default normalization (deterministic)
    pred_text: ASR prediction
    verbose: set to True to output intermediate output of adjustments (for debugging)

    Return:
        semiotic_spans: List[str] - List of semiotic spans from raw text
        pred_texts: List[str] - List of pred_texts correponding to semiotic_spans
        norm_spans: List[str] - List of normalized texts correponding to semiotic_spans
        raw_text_masked_list: List[str] - List of words from raw text where every semiotic span is replaces with SEMIOTIC_TAG
        raw_text_mask_idx: List[int] - List of indexes of SEMIOTIC_TAG in raw_text_masked_list

        e.g.:
            >>> raw = 'This is #4 ranking on G.S.K.T.'
            >>> pred_text = 'this iss for ranking on g k p'
            >>> norm = 'This is nubmer four ranking on GSKT'

            output:
            semiotic_spans: ['is #4', 'G.S.K.T.']
            pred_texts: ['iss for', 'g k p']
            norm_spans: ['is nubmer four', 'GSKT']
            raw_text_masked_list: ['This', '[SEMIOTIC_SPAN]', 'ranking', 'on', '[SEMIOTIC_SPAN]']
            raw_text_mask_idx: [1, 4]
    """
    pass


def get_alignment(raw: str, norm: str, pred_text: str, verbose: bool = False):
    """
    Aligns raw text with deterministically normalized text and ASR output, finds semiotic spans
    """
    pass


if __name__ == "__main__":
    raw = 'This is a #4 ranking on G.S.K.T.'
    pred_text = 'this iss p k for ranking on g k p'
    norm = 'This is nubmer four ranking on GSKT'

    output = get_alignment(raw, norm, pred_text, True)
    print(output)
