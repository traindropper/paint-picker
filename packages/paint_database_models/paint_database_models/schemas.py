from pydantic import BaseModel
from paint_database_models.base_classes import ManufacturerEnum, FinishEnum, PaintMediumEnum


class PaintBase(BaseModel):
    color: str
    manufacturer: ManufacturerEnum
    finish: FinishEnum
    medium: PaintMediumEnum
    swatch: str
    quantity: int


class PaintCreate(PaintBase):
    pass


class PaintUpdate(BaseModel):
    color: str | None = None
    manufacturer: ManufacturerEnum | None = None
    finish: FinishEnum | None = None
    medium: PaintMediumEnum | None = None
    swatch: str | None = None
    quantity: int | None = None


class PaintOut(PaintBase):
    id: int

    model_config = {"from_attributes": True}
