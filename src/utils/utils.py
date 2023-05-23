from typing import Optional, Dict, Tuple

def dict_to_list(
    dictionary: Dict[int or str, str], cased: Optional[bool]=False) -> Tuple[list[str], list[str]]:
    keys , vals = list(map(str, list(dictionary.keys()))), list(dictionary.values())
    if cased:
        return keys, vals
    return keys, list(map(lambda x: str(x).lower(), vals))
