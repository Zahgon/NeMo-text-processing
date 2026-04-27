# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from argparse import ArgumentParser
from typing import List

import regex as re

from nemo_text_processing.text_normalization.data_loader_utils import (
    EOS_TYPE,
    Instance,
    load_files,
    training_data_to_sentences,
)

"""
This file is for evaluation purposes.
filter_loaded_data() cleans data (list of instances) for text normalization. Filters and cleaners can be specified for each semiotic class individually.
For example, normalized text should only include characters and whitespace characters but no punctuation. 
            Cardinal unnormalized instances should contain at least one integer and all other characters are removed.
"""


class Filter:
    """
    Filter class

    Args:
        class_type: semiotic class used in dataset
        process_func: function to transform text
        filter_func:  function to filter text

    """

    def __init__(self, class_type: str, process_func: object, filter_func: object):
        self.class_type = class_type
        self.process_func = process_func
        self.filter_func = filter_func

    def filter(self, instance: Instance) -> bool:
        """
        filter function

        Args:
            filters given instance with filter function

        Returns: True if given instance fulfills criteria or does not belong to class type
        """
        pass

    def process(self, instance: Instance) -> Instance:
        """
        process function

        Args:
            processes given instance with process function

        Returns: processed instance if instance belongs to expected class type or original instance
        """
        pass


def filter_cardinal_1(instance: Instance) -> bool:
    pass


def process_cardinal_1(instance: Instance) -> Instance:
    pass


def filter_ordinal_1(instance: Instance) -> bool:
    pass


def process_ordinal_1(instance: Instance) -> Instance:
    pass


def filter_decimal_1(instance: Instance) -> bool:
    pass


def process_decimal_1(instance: Instance) -> Instance:
    pass


def filter_measure_1(instance: Instance) -> bool:
    pass


def process_measure_1(instance: Instance) -> Instance:
    pass


def filter_money_1(instance: Instance) -> bool:
    pass


def process_money_1(instance: Instance) -> Instance:
    pass


def filter_time_1(instance: Instance) -> bool:
    pass


def process_time_1(instance: Instance) -> Instance:
    pass


def filter_plain_1(instance: Instance) -> bool:
    pass


def process_plain_1(instance: Instance) -> Instance:
    pass


def filter_punct_1(instance: Instance) -> bool:
    pass


def process_punct_1(instance: Instance) -> Instance:
    pass


def filter_date_1(instance: Instance) -> bool:
    pass


def process_date_1(instance: Instance) -> Instance:
    pass


def filter_letters_1(instance: Instance) -> bool:
    pass


def process_letters_1(instance: Instance) -> Instance:
    pass


def filter_verbatim_1(instance: Instance) -> bool:
    pass


def process_verbatim_1(instance: Instance) -> Instance:
    pass


def filter_digit_1(instance: Instance) -> bool:
    pass


def process_digit_1(instance: Instance) -> Instance:
    pass


def filter_telephone_1(instance: Instance) -> bool:
    pass


def process_telephone_1(instance: Instance) -> Instance:
    pass


def filter_electronic_1(instance: Instance) -> bool:
    pass


def process_electronic_1(instance: Instance) -> Instance:
    pass


def filter_fraction_1(instance: Instance) -> bool:
    pass


def process_fraction_1(instance: Instance) -> Instance:
    pass


def filter_address_1(instance: Instance) -> bool:
    pass


def process_address_1(instance: Instance) -> Instance:
    pass


filters = []
filters.append(Filter(class_type="CARDINAL", process_func=process_cardinal_1, filter_func=filter_cardinal_1))
filters.append(Filter(class_type="ORDINAL", process_func=process_ordinal_1, filter_func=filter_ordinal_1))
filters.append(Filter(class_type="DECIMAL", process_func=process_decimal_1, filter_func=filter_decimal_1))
filters.append(Filter(class_type="MEASURE", process_func=process_measure_1, filter_func=filter_measure_1))
filters.append(Filter(class_type="MONEY", process_func=process_money_1, filter_func=filter_money_1))
filters.append(Filter(class_type="TIME", process_func=process_time_1, filter_func=filter_time_1))

filters.append(Filter(class_type="DATE", process_func=process_date_1, filter_func=filter_date_1))
filters.append(Filter(class_type="PLAIN", process_func=process_plain_1, filter_func=filter_plain_1))
filters.append(Filter(class_type="PUNCT", process_func=process_punct_1, filter_func=filter_punct_1))
filters.append(Filter(class_type="LETTERS", process_func=process_letters_1, filter_func=filter_letters_1))
filters.append(Filter(class_type="VERBATIM", process_func=process_verbatim_1, filter_func=filter_verbatim_1))
filters.append(Filter(class_type="DIGIT", process_func=process_digit_1, filter_func=filter_digit_1))
filters.append(Filter(class_type="TELEPHONE", process_func=process_telephone_1, filter_func=filter_telephone_1))
filters.append(Filter(class_type="ELECTRONIC", process_func=process_electronic_1, filter_func=filter_electronic_1))
filters.append(Filter(class_type="FRACTION", process_func=process_fraction_1, filter_func=filter_fraction_1))
filters.append(Filter(class_type="ADDRESS", process_func=process_address_1, filter_func=filter_address_1))
filters.append(Filter(class_type=EOS_TYPE, process_func=lambda x: x, filter_func=lambda x: True))


def filter_loaded_data(data: List[Instance], verbose: bool = False) -> List[Instance]:
    """
    Filters list of instances

    Args:
        data: list of instances

    Returns: filtered and transformed list of instances
    """
    pass


def parse_args():
    pass


if __name__ == "__main__":
    args = parse_args()
    file_path = args.input

    print("Loading training data: " + file_path)
    instance_list = load_files([file_path])  # List of instances
    filtered_instance_list = filter_loaded_data(instance_list, args.verbose)
    training_data_to_sentences(filtered_instance_list)
