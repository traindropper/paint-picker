from typing import Any

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