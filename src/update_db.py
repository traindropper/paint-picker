from typing import Any
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import Session, Query
import logging

from src.models import Manufacturer, Finish, PaintMedium, PaintDTO, PaintUpdateDTO, Paint
from src.database_helpers import add_com_ref, normalize_string

LOGGER: logging.Logger = logging.getLogger(__name__)


def get_or_create_manufacturer(session: Session, name: str) -> Manufacturer:
    normalized_name: str = normalize_string(name)
    obj: Manufacturer | None = session.query(Manufacturer).filter_by(normalized_name=normalized_name).first()
    if not obj:
        obj = Manufacturer(name=name, normalized_name=normalized_name)
        add_com_ref(session, obj)
    return obj

def get_or_create_finish(session: Session, name: str | None = None) -> Finish | None: 
    normalized_name: str = normalize_string(name)
    if not name:
        return None
    obj: Finish | None = session.query(Finish).filter_by(normalized_name=normalized_name).first()
    if not obj:
        obj = Finish(name=name, normalized_name=normalized_name)
        add_com_ref(session, obj)
    return obj

def get_or_create_paint_medium(session: Session, name: str | None = None) -> PaintMedium | None: 
    if not name:
        return None
    normalized_name: str = normalize_string(name)
    obj: PaintMedium | None = session.query(PaintMedium).filter_by(normalized_name=normalized_name).first()
    if not obj:
        obj = PaintMedium(name=name, normalized_name=normalized_name)
        add_com_ref(session, obj)
    return obj

def upsert_paint(session: Session, paint_data: PaintDTO) -> Paint:
    manufacturer: Manufacturer = get_or_create_manufacturer(session, paint_data.manufacturer)
    finish: Finish | None = get_or_create_finish(session, paint_data.finish)
    paint_medium: PaintMedium | None = get_or_create_paint_medium(session, paint_data.paint_medium)

    conditions_list: list[Any] = [
        Paint.manufacturer_id == manufacturer.id,
        Paint.normalized_color == normalize_string(paint_data.color),
    ]
    if finish:
        conditions_list.append(Paint.finish_id == finish.id)
    if paint_medium:
        conditions_list.append(Paint.paint_medium_id == paint_medium.id)

    paint: Paint | None = session.query(Paint).filter(*conditions_list).first()

    # to adjust quantities by more than one unit at a time, directly call the adjust function
    if paint:  # Since the paint already exists, merely increment it by one.
        adjust_paint_quantity(session, paint, delta=1)
        return paint

    paint = Paint(
        manufacturer_id=manufacturer.id,
        color=paint_data.color,
        normalized_color=normalize_string(paint_data.color),
        finish_id=finish.id if finish else None,
        paint_medium_id=paint_medium.id if paint_medium else None,
        quantity_owned=1
    )
    add_com_ref(session, paint)
    LOGGER.info("Created new paint with ID: %s", paint.id)
    LOGGER.info("It has this data: %s", paint_data)
    return paint

def delete_paint(session: Session, paint_data: PaintDTO) -> bool:
    manufacturer: Manufacturer | None = session.query(Manufacturer).filter_by(normalized_name=normalize_string(paint_data.manufacturer)).first()
    if manufacturer is None:
        return False
    finish: Finish | None = session.query(Finish).filter_by(normalized_name=normalize_string(paint_data.finish)).first() if paint_data.finish else None
    paint_medium: PaintMedium | None = session.query(PaintMedium).filter_by(normalized_name=normalize_string(paint_data.paint_medium)).first() if paint_data.paint_medium else None
    paint: Paint | None = session.query(Paint).filter_by(
        manufacturer_id=manufacturer.id,
        color=paint_data.color,
        normalized_color=normalize_string(paint_data.color),
        finish_id=finish.id if finish else None,
        paint_medium=paint_medium.id if paint_medium else None,
    ).first()
    if not paint:
        return False
    session.delete(paint)
    session.commit()
    LOGGER.info("Deleted paint: %s", paint.id)
    LOGGER.info("It had this data: %s", paint)
    return True

def adjust_paint_quantity(
    session: Session,
    paint_data: PaintDTO | Paint,
    delta: int
) -> bool:
    """
    Adjust the quantity_owned of a paint, deletes if quantity is zero.
    """
    paint: Paint | None
    if isinstance(paint_data, PaintDTO):
        manufacturer_name: str = paint_data.manufacturer
        finish_name: str | None = paint_data.finish
        paint_medium_name: str | None = paint_data.paint_medium

        manufacturer: Manufacturer | None = session.query(Manufacturer).filter_by(
            normalized_name=normalize_string(manufacturer_name)
        ).first()
        if not manufacturer:
            return None
        
        finish: Finish | None = None
        if finish_name:
            finish = session.query(Finish).filter_by(
                normalized_name = normalize_string(finish_name)
            ).first()
        
        paint_medium: PaintMedium | None = None
        if paint_medium_name:
            paint_medium = session.query(PaintMedium).filter_by(
                normalized_name=normalize_string(paint_medium_name)
            ).first()
        
        conditions_list: list[Any] = [
            Paint.manufacturer_id == manufacturer.id,
            Paint.normalized_color == normalize_string(paint_data.color),
        ]
        if finish:
            conditions_list.append(Paint.finish_id == finish.id)
        if paint_medium:
            conditions_list.append(Paint.paint_medium_id == paint_medium.id)

        paint: Paint | None = session.query(Paint).filter(*conditions_list).first()

        if not paint:
            return False
    
    else:
        paint = paint_data
    
    LOGGER.info("INCREMENTING")
    paint.quantity_owned += delta

    LOGGER.info("Incremented quantity of paint: %s by %s", paint.id, delta)

    if paint.quantity_owned <= 0:
        LOGGER.info("Quantity is <= 0, deleting...")
        session.delete(paint)
        session.commit()
    else:
        LOGGER.info("New quantity = %s", paint.quantity_owned)
        session.commit()
    
    return True