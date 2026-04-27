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

import string
from collections import OrderedDict
from typing import Dict, List, Union

PRESERVE_ORDER_KEY = "preserve_order"
EOS = "<EOS>"


class TokenParser:
    """
    Parses tokenized/classified text, e.g. 'tokens { money { integer: "20" currency: "$" } } tokens { name: "left"}'

    Args
        text: tokenized text
    """

    def __call__(self, text):
        """
        Setup function

        Args:
            text: text to be parsed

        """
        self.text = text
        self.len_text = len(text)
        self.char = text[0]  # cannot handle empty string
        self.index = 0

    def parse(self) -> List[dict]:
        """
        Main function. Implements grammar:
        A -> space F space F space F ... space

        Returns list of dictionaries
        """
        pass

    def parse_token(self) -> Dict[str, Union[str, dict]]:
        """
        Implements grammar:
        F-> no_space KG no_space

        Returns: K, G as dictionary values
        """
        pass

    def parse_token_value(self) -> Union[str, dict]:
        """
        Implements grammar:
        G-> no_space :"VALUE" no_space | no_space {A} no_space

        Returns: string or dictionary
        """
        pass

    def parse_char(self, exp) -> bool:
        """
        Parses character

        Args:
            exp: character to read in

        Returns true if successful
        """
        pass

    def parse_chars(self, exp) -> bool:
        """
        Parses characters

        Args:
            exp: characters to read in

        Returns true if successful
        """
        pass

    def parse_string_key(self) -> str:
        """
        Parses string key, can only contain ascii and '_' characters

        Returns parsed string key
        """
        pass

    def parse_string_value(self) -> str:
        """
        Parses string value, ends with quote followed by space

        Returns parsed string value
        """
        pass

    def parse_ws(self):
        """
        Deletes whitespaces.

        Returns true if not EOS after parsing
        """
        pass

    def read(self):
        """
        Reads in next char.

        Returns true if not EOS
        """
        pass
