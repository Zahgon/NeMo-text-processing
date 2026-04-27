# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import json
import os
from argparse import ArgumentParser
from time import perf_counter
from typing import List, Optional, Tuple

import editdistance
import pynini
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.data_loader_utils import post_process_punct, pre_process
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.utils_audio_based import get_alignment
from nemo_text_processing.utils.logging import logger

"""
The script provides multiple normalization options and chooses the best one that minimizes CER of the ASR output
(most of the semiotic classes use deterministic=False flag).

To run this script with a .json manifest file, the manifest file should contain the following fields:
    "text" - raw text (could be changed using "--manifest_text_field")
    "pred_text" - ASR model prediction, see https://github.com/NVIDIA/NeMo/blob/main/examples/asr/transcribe_speech.py 
        on how to transcribe manifest
    
    Example for a manifest line:
        {"text": "In December 2015, ...", "pred_txt": "on december two thousand fifteen"}

    When the manifest is ready, run:
        python normalize_with_audio.py \
            --manifest PATH/TO/MANIFEST.JSON \
            --language en \
            --output_filename=<PATH TO OUTPUT .JSON MANIFEST> \
            --n_jobs=-1 \
            --batch_size=300 \
            --manifest_text_field="text"

To see possible normalization options for a text input without an audio file (could be used for debugging), run:
    python python normalize_with_audio.py --text "RAW TEXT"

Specify `--cache_dir` to generate .far grammars once and re-used them for faster inference
"""


class NormalizerWithAudio(Normalizer):
    """
    Normalizer class that converts text from written to spoken form.
    Useful for TTS preprocessing.

    Args:
        input_case: expected input capitalization
        lang: language
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
        post_process: WFST-based post processing, e.g. to remove extra spaces added during TN.
            Note: punct_post_process flag in normalize() supports all languages.
        max_number_of_permutations_per_split: a maximum number
                of permutations which can be generated from input sequence of tokens.
    """

    def __init__(
        self,
        input_case: str,
        lang: str = 'en',
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
        lm: bool = False,
        post_process: bool = True,
        max_number_of_permutations_per_split: int = 729,
    ):

        # initialize non-deterministic normalizer
        super().__init__(
            input_case=input_case,
            lang=lang,
            deterministic=False,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            whitelist=whitelist,
            lm=lm,
            post_process=post_process,
        )
        self.tagger_non_deterministic = self.tagger
        self.verbalizer_non_deterministic = self.verbalizer

        if lang != "ru":
            # initialize deterministic normalizer
            super().__init__(
                input_case=input_case,
                lang=lang,
                deterministic=True,
                cache_dir=cache_dir,
                overwrite_cache=overwrite_cache,
                whitelist=whitelist,
                lm=lm,
                post_process=post_process,
                max_number_of_permutations_per_split=max_number_of_permutations_per_split,
            )
        else:
            self.tagger, self.verbalizer = None, None
        self.lm = lm

    def normalize(
        self,
        text: str,
        n_tagged: int,
        punct_post_process: bool = True,
        verbose: bool = False,
        pred_text: Optional[str] = None,
        cer_threshold: float = -1,
        **kwargs,
    ) -> str:
        """
        Main function. Normalizes tokens from written to spoken form
            e.g. 12 kg -> twelve kilograms

        Args:
            text: string that may include semiotic classes
            n_tagged: number of tagged options to consider, -1 - to get all possible tagged options
            punct_post_process: whether to normalize punctuation
            verbose: whether to print intermediate meta information
            pred_text: ASR model transcript
            cer_threshold: if CER for pred_text and the normalization option is above the cer_threshold,
                default deterministic normalization will be used. Set to -1 to disable cer-based filtering.
                Specify the value in %, e.g. 100 not 1.

        Returns:
            normalized text options (usually there are multiple ways of normalizing a given semiotic class)
        """
        pass

    def normalize_non_deterministic(
        self, text: str, n_tagged: int, punct_post_process: bool = True, verbose: bool = False
    ):
        # get deterministic option
        pass

    def normalize_line(
        self,
        n_tagged: int,
        line: str,
        verbose: bool = False,
        punct_pre_process=False,
        punct_post_process=True,
        text_field: str = "text",
        asr_pred_field: str = "pred_text",
        output_field: str = "normalized",
        cer_threshold: float = -1,
    ):
        """
        Normalizes "text_field" in line from a .json manifest

        Args:
            n_tagged: number of normalization options to return
            line: line of a .json manifest
            verbose: set to True to see intermediate output of normalization
            punct_pre_process: set to True to do punctuation pre-processing
            punct_post_process: set to True to do punctuation post-processing
            text_field: name of the field in the manifest to normalize
            asr_pred_field: name of the field in the manifest with ASR predictions
            output_field: name of the field in the manifest to save normalized text
            cer_threshold: if CER for pred_text and the normalization option is above the cer_threshold,
                default deterministic normalization will be used. Set to -1 to disable cer-based filtering.
                Specify the value in %, e.g. 100 not 1.
        """
        pass

    def _get_tagged_text(self, text, n_tagged):
        """
        Returns text after tokenize and classify
        Args;
            text: input  text
            n_tagged: number of tagged options to consider, -1 - return all possible tagged options
        """
        pass

    def _verbalize(self, tagged_text: str, normalized_texts: List[str], n_tagged: int, verbose: bool = False):
        """
        Verbalizes tagged text

        Args:
            tagged_text: text with tags
            normalized_texts: list of possible normalization options
            verbose: if true prints intermediate classification results
        """
        pass

    def select_best_match(
        self,
        normalized_texts: List[str],
        pred_text: str,
        verbose: bool = False,
        remove_punct: bool = False,
    ):
        """
        Selects the best normalization option based on the lowest CER

        Args:
            normalized_texts: normalized text options
            pred_text: ASR model transcript of the audio file corresponding to the normalized text
            verbose: whether to print intermediate meta information
            remove_punct: whether to remove punctuation before calculating CER

        Returns:
            normalized text with the lowest CER and CER value
        """
        pass


def calculate_cer(normalized_texts: List[str], pred_text: str, remove_punct=False) -> List[Tuple[str, float]]:
    """
    Calculates character error rate (CER)

    Args:
        normalized_texts: normalized text options
        pred_text: ASR model output

    Returns: normalized options with corresponding CER
    """
    pass


def parse_args():
    pass


if __name__ == "__main__":
    args = parse_args()

    args.whitelist = os.path.abspath(args.whitelist) if args.whitelist else None
    if args.text is not None:
        normalizer = NormalizerWithAudio(
            input_case=args.input_case,
            lang=args.language,
            cache_dir=args.cache_dir,
            overwrite_cache=args.overwrite_cache,
            whitelist=args.whitelist,
            lm=args.lm,
            max_number_of_permutations_per_split=args.max_number_of_permutations_per_split,
        )
        start = perf_counter()
        if os.path.exists(args.text):
            with open(args.text, 'r') as f:
                args.text = f.read().strip()

        options = normalizer.normalize(
            text=args.text,
            n_tagged=args.n_tagged,
            punct_post_process=not args.no_punct_post_process,
            verbose=args.verbose,
        )
        for option in options:
            logger.info(option)
    elif args.manifest.endswith('.json'):
        normalizer = NormalizerWithAudio(
            input_case=args.input_case,
            lang=args.language,
            cache_dir=args.cache_dir,
            overwrite_cache=args.overwrite_cache,
            whitelist=args.whitelist,
            max_number_of_permutations_per_split=args.max_number_of_permutations_per_split,
        )
        start = perf_counter()
        normalizer.normalize_manifest(
            manifest=args.manifest,
            n_jobs=args.n_jobs,
            punct_pre_process=True,
            punct_post_process=not args.no_punct_post_process,
            batch_size=args.batch_size,
            output_filename=args.output_filename,
            n_tagged=args.n_tagged,
            text_field=args.manifest_text_field,
            asr_pred_field=args.manifest_asr_pred_field,
            cer_threshold=args.cer_threshold,
            verbose=args.verbose,
        )
    else:
        raise ValueError(
            "Provide either path to .json manifest with '--manifest' OR "
            + "an input text with '--text' (for debugging without audio)"
        )
    logger.info(f'Execution time: {round((perf_counter() - start)/60, 2)} min.')
