from typing import List


def _split(sentences: List[str], delimiter: str, max_len: int, min_len: int):
    """
    Splits sentences based by the specified delimiter. Will attempt to split and combine sentences to get target
        min/max length.

    Args:
        sentences: Sentences to split into segments.
        delimiter: symbol to split by
        max_len: the maximum number of symbols in the output sentences (the result will be the closest len match)
        min_len: the minimum number of the output sentences (the result will be the closest len match)
    """
    pass


def additional_split(sentences: List[str], split_on_symbols: str, max_len: int = 1000, min_len: int = 2) -> List[str]:
    """
    Splits sentences by split_on_symbols.

    Args:
        sentences: Sentences to split into segments.
        split_on_symbols: Symbols to split sentences if eos sentence split resulted in a long sequence.
            Use '|' as a separator between symbols, for example: ';|:| ', will attempt to split each sentence
            by semi-colon ";", colon ":", and space " ".
        max_len: the maximum number of symbols in the output sentences (the result will be the closest len match)
        min_len: the minimum number of the output sentences (the result will be the closest len match)
    """
    pass
