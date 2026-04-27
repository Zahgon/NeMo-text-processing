# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, NEMO_SPACE
from nemo_text_processing.text_normalization.es import LOCALIZATION
from nemo_text_processing.text_normalization.es.utils import get_abs_path, load_labels

digits = pynini.project(pynini.string_file(get_abs_path("data/numbers/digit.tsv")), "input")
tens = pynini.project(pynini.string_file(get_abs_path("data/numbers/ties.tsv")), "input")
teens = pynini.project(pynini.string_file(get_abs_path("data/numbers/teen.tsv")), "input")
twenties = pynini.project(pynini.string_file(get_abs_path("data/numbers/twenties.tsv")), "input")
hundreds = pynini.project(pynini.string_file(get_abs_path("data/numbers/hundreds.tsv")), "input")

accents = pynini.string_map([("Ã¡", "a"), ("Ã©", "e"), ("Ã­", "i"), ("Ã³", "o"), ("Ãº", "u")])

if LOCALIZATION == "am":  # Setting localization for central and northern america formatting
    cardinal_separator = pynini.string_map([",", NEMO_SPACE])
    decimal_separator = pynini.accep(".")
else:
    cardinal_separator = pynini.string_map([".", NEMO_SPACE])
    decimal_separator = pynini.accep(",")

ones = pynini.union("un", "Ãºn")
fem_ones = pynini.union(pynini.cross("un", "una"), pynini.cross("Ãºn", "una"), pynini.cross("uno", "una"))
one_to_one_hundred = pynini.union(digits, "uno", tens, teens, twenties, tens + pynini.accep(" y ") + digits)
fem_hundreds = hundreds @ pynini.cdrewrite(pynini.cross("ientos", "ientas"), "", "", NEMO_SIGMA)


ES_MINUS = pynini.union("menos", "Menos", "MENOS").optimize()
ES_PLUS = pynini.union("mÃ¡s", "MÃ¡s", "MÃ�S").optimize()


def strip_accent(fst: "pynini.FstLike") -> "pynini.FstLike":
    """
    Converts all accented vowels to non-accented equivalents

    Args:
        fst: Any fst. Composes vowel conversion onto fst's output strings
    """
    pass


def shift_cardinal_gender(fst: "pynini.FstLike") -> "pynini.FstLike":
    """
    Applies gender conversion rules to a cardinal string. These include: rendering all masculine forms of "uno" (including apocopated forms) as "una" and
    Converting all gendered numbers in the hundreds series (200,300,400...) to feminine equivalent (e.g. "doscientos" -> "doscientas"). Conversion only applies
    to value place for <1000 and multiple of 1000. (e.g. "doscientos mil doscientos" -> "doscientas mil doscientas".) For place values greater than the thousands, there
    is no gender shift as the higher powers of ten ("millones", "billones") are masculine nouns and any conversion would be formally
    ungrammatical.
    e.g.
        "doscientos" -> "doscientas"
        "doscientos mil" -> "doscientas mil"
        "doscientos millones" -> "doscientos millones"
        "doscientos mil millones" -> "doscientos mil millones"
        "doscientos millones doscientos mil doscientos" -> "doscientos millones doscientas mil doscientas"

    Args:
        fst: Any fst. Composes conversion onto fst's output strings
    """
    pass


def shift_number_gender(fst: "pynini.FstLike") -> "pynini.FstLike":
    """
    Performs gender conversion on all verbalized numbers in output. All values in the hundreds series (200,300,400) are changed to
    feminine gender (e.g. "doscientos" -> "doscientas") and all forms of "uno" (including apocopated forms) are converted to "una".
    This has no boundary restriction and will perform shift across all values in output string.
    e.g.
        "doscientos" -> "doscientas"
        "doscientos millones" -> "doscientas millones"
        "doscientos millones doscientos" -> "doscientas millones doscientas"

    Args:
        fst: Any fst. Composes conversion onto fst's output strings
    """
    pass


def strip_cardinal_apocope(fst: "pynini.FstLike") -> "pynini.FstLike":
    """
    Reverts apocope on cardinal strings in line with formation rules. e.g. "un" -> "uno". Due to cardinal formation rules, this in effect only
    affects strings where the final value is a variation of "un".
    e.g.
        "un" -> "uno"
        "veintiÃºn" -> "veintiuno"

    Args:
        fst: Any fst. Composes conversion onto fst's output strings
    """
    pass


def add_cardinal_apocope_fem(fst: "pynini.FstLike") -> "pynini.FstLike":
    """
    Adds apocope on cardinal strings in line with stressing rules. e.g. "una" -> "un". This only occurs when "una" precedes a stressed "a" sound in formal speech. This is not predictable
    with text string, so is included for non-deterministic cases.
    e.g.
        "una" -> "un"
        "veintiuna" -> "veintiun"

    Args:
        fst: Any fst. Composes conversion onto fst's output strings
    """
    pass


def roman_to_int(fst: "pynini.FstLike") -> "pynini.FstLike":
    """
    Alters given fst to convert Roman integers (lower and upper cased) into Arabic numerals. Valid for values up to 1000.
    e.g.
        "V" -> "5"
        "i" -> "1"

    Args:
        fst: Any fst. Composes fst onto Roman conversion outputs.
    """
    pass
