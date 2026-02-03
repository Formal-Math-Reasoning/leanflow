from typing import Optional, Union
import re

from ..utils import Pos

def _parse_pos_from_string(data: Optional[Union[str, Pos]]) -> Optional[Pos]:
    """A dacite type hook to convert a string like '(l, c)' into a Pos object.
    
    Prevents deserialization errors when data is re-processed.

    Args:
        data (Optional[Union[str, Pos]]): The input data, either a string or a Pos object.

    Returns:
        Optional[Pos]: The parsed Pos object, or None if input is empty/invalid.
    """
    if isinstance(data, Pos):
        return data

    if not data:
        return None
    
    # parse the string
    match = re.match(r'\((\d+),\s*(\d+)\)', data)
    if match:
        line, column = map(int, match.groups())
        return Pos(line=line, column=column)
    return None