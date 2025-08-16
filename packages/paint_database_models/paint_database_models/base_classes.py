from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum
import json


class ManufacturerEnum(str, Enum):
    MR_HOBBY = "Mr. Hobby"
    UNKNOWN = "Unknown"


class FinishEnum(str, Enum):
    MATTE = "Matte"
    GLOSS = "Gloss"
    SEMI_GLOSS = "Semi-gloss"
    PEARL = "Pearl"
    CLEAR = "Clear"
    METALLIC = "Metallic"
    FLOURESCENT = "Flourescent"
    UNKNOWN = "Unknown"


class PaintMediumEnum(str, Enum):
    LACQUER = "Lacquer"
    Acrylic = "Acrylic"
    ENAMEL = "Enamel"
    OIL = "Oil"
    UNKNOWN = "Unknown"


class DataclassEnumEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Enum):
            return o.value
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)
