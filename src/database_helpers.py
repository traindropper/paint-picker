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

def sync_enum_to_table(
    session: Session,
    enum_class: Type[T],
    model_class,
    name_field: str = "name",
    normalized_field: str = "normalized_name"
) -> None:
    """
    Ensures every enum value exists as a row in the reference table.
    session (Session): The SQLAlchemy session.
    enum_class (Type[Enum]): The enum class to sync to the table.
    model_class (????): The model class which corresponds to the enum.
    name_field (str): Name of the field to store the enum value in.
    normalized_field (str): Name of the field to store a normalized value in.
    """
    for enum_val in enum_class:
        exists = session.query(model_class).filter(
            getattr(model_class, name_field) == enum_val.value
        ).first()
        if not exists:
            new_row = model_class(
                **{
                    name_field: enum_val.value,
                    normalized_field: normalize_string(enum_val.value)
                }
            )
            session.add(new_row)
    session.commit()