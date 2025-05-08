def unit_tag(unitTagIndex: int, unitTagRecycle: int) -> int:
    return (unitTagIndex << 18) + unitTagRecycle


def unit_tag_index(unitTag: int) -> int:
    return (unitTag >> 18) & 0x00003FFF


def unit_tag_recycle(unitTag: int) -> int:
    return (unitTag) & 0x0003FFFF
