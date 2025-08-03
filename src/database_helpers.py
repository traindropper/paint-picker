from typing import Any
from enum import Enum
from typing import Type, TypeVar

T = TypeVar('T', bound=Enum)

from sqlalchemy.orm import Session, Query
import re


def add_com_ref(session: Session, obj: Any) -> None:
    """Add, commit, refresh with a given object"""
    session.add(obj)
    session.commit()
    session.refresh(obj)

def normalize_string(s: str) -> str:
    s=s.lower()
    return re.sub(f"[\W_]+", "", s)

def normalize_to_enum(value: str, enum_class: Type[T]) -> T:
    """Normalize a string to an Enum value, returning UNKNOWN if not found."""
    normalized_value = normalize_string(value)
    for enum_member in enum_class:
        if normalize_string(enum_member.value) == normalized_value:
            return enum_member
    return enum_class.UNKNOWN  # Return UNKNOWN if no match found