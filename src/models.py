from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column, relationship, DeclarativeBase, Session
from sqlalchemy import String, Integer, ForeignKey
from src.base_classes import ManufacturerEnum, PaintMediumEnum, FinishEnum
from src.database_helpers import normalize_string
from dataclasses import dataclass


class Base(DeclarativeBase):
    pass

class Manufacturer(Base):
    __tablename__ = "manufacturers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    normalized_name: Mapped[str] = mapped_column(String, unique=True, nullable=False)

    paints: Mapped[list["Paint"]] = relationship(back_populates="manufacturer")

class Finish(Base):
    __tablename__ = "finishes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    normalized_name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    
    paints: Mapped[list["Paint"]] = relationship(back_populates="finish")


class PaintMedium(Base):
    __tablename__ = "paint_mediums"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    normalized_name: Mapped[str] = mapped_column(String, unique=True, nullable=False)

    paints: Mapped[list["Paint"]] = relationship(back_populates="paint_medium")

class Paint(Base):
    __tablename__ = "paints"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    color: Mapped[str] = mapped_column(String, nullable=False)
    normalized_color: Mapped[str] = mapped_column(String, nullable=False)
    swatch: Mapped[str] = mapped_column(String, nullable=False)
    manufacturer_id: Mapped[int] = mapped_column(ForeignKey("manufacturers.id"), nullable=False)
    finish_id: Mapped[Optional[int]] = mapped_column(ForeignKey("finishes.id"), nullable=True)
    paint_medium_id: Mapped[Optional[int]] = mapped_column(ForeignKey("paint_mediums.id"), nullable=True)
    quantity_owned: Mapped[int] = mapped_column(Integer, default=0)

    manufacturer: Mapped["Manufacturer"] = relationship(back_populates="paints")
    finish: Mapped[Optional["Finish"]] = relationship(back_populates="paints")
    paint_medium: Mapped[Optional["PaintMedium"]] = relationship(back_populates="paints")


@dataclass
class PaintDTO:
    manufacturer: ManufacturerEnum 
    color: str
    swatch: str | None
    finish: FinishEnum | None = None
    paint_medium: PaintMediumEnum | None = None


@dataclass
class PaintUpdateDTO:
    id: int
    manufacturer: ManufacturerEnum 
    color: str
    swatch: str
    finish: FinishEnum | None = None
    paint_medium: PaintMediumEnum | None = None


def paint_to_dto(paint: Paint) -> PaintUpdateDTO:

    return PaintUpdateDTO(
        color=paint.color,
        id=paint.id,
        swatch=paint.swatch,
        manufacturer=ManufacturerEnum(paint.manufacturer.name),
        finish=FinishEnum(paint.finish.name) if paint.finish else None,
        paint_medium=PaintMediumEnum(paint.paint_medium.name) if paint.paint_medium else None,
    )


def create_paint_from_dto(dto: PaintDTO, session: Session) -> Paint:
    return Paint(
        color=dto.color,
        normalized_color=normalize_string(dto.color),
        swatch=dto.swatch,
        manufacturer=session.query(Manufacturer).filter_by(normalized_name=normalize_string(dto.manufacturer.value)).one(),
        finish=session.query(Finish).filter_by(normalized_name=normalize_string(dto.finish.value)).one() if dto.finish else None,
        paint_medium=session.query(PaintMedium).filter_by(normalized_name=normalize_string(dto.paint_medium.value)).one() if dto.paint_medium else None,
    )

