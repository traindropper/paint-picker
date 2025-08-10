from pathlib import Path
import shutil
from typing import Any
from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, Engine, select
from src.models import Base, Paint, PaintDTO, Manufacturer, Finish, PaintMedium
from src.database_helpers import add_com_ref
from src.paint_parser import parse_image_as_string
from src.update_db import upsert_paint, get_or_create_finish, get_or_create_manufacturer, get_or_create_paint_medium
from src.database_helpers import normalize_to_enum
from src.base_classes import FinishEnum, PaintMediumEnum, ManufacturerEnum
from enum import Enum
from src.schemas import PaintOut, PaintUpdate
from src.database_helpers import normalize_string
from typing import Annotated

# Dependency on DB session
DATABASE_URL: str = "sqlite:///./paintdb.sqlite3"
engine: Engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(engine)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

UPLOAD_DIR: Path = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OCR_DIR: Path = Path("static/ocr")
OCR_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
templates= Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
session_state: dict = {} # Placeholder for session management

@app.get("/", response_class=HTMLResponse, name="home")
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/paints")
async def get_paints(request: Request, db: Session = Depends(get_db)):
    paint_list: list[dict[str, Any]] = [] 
    paints: list[Paint] = db.query(Paint).all()
    for paint in paints:
        paint_data: dict[str, Any] = {
            "id": paint.id,
            "color": paint.color,
            "swatch": paint.swatch,
            "manufacturer": paint.manufacturer.name,
            "finish": paint.finish.name if paint.finish else None,
            "medium": paint.paint_medium.name if paint.paint_medium else None,
            "quantity": paint.quantity_owned
        }
        paint_list.append(paint_data)
    return paint_list

@app.get("/paints/{paint_id}", response_model=PaintOut)
def get_paint(paint_id: int, db: Session = Depends(get_db)):
    paint: Paint | None = db.query(Paint).filter(Paint.id==paint_id).first()
    if not paint:
        raise HTTPException(status_code=404, detail="Paint not found.")
    return Paint

@app.put("/paints/{paint_id}", response_model=PaintOut)
def update_paint(paint_id: int, update: PaintUpdate, db: Session = Depends(get_db)):
    paint_item: Paint | None = db.query(Paint).filter(Paint.id==paint_id).first()
    if not paint_item:
        raise HTTPException(status_code=404, detail="Paint not found.")
    update_data: dict = update.model_dump(exclude_unset=True)
    
    manufacturer: Manufacturer = get_or_create_manufacturer(db, update_data["manufacturer"].value)
    finish: Finish = get_or_create_finish(db, update_data["finish"].value)
    paint_medium: PaintMedium = get_or_create_paint_medium(db, update_data["medium"].value)

    paint_item.manufacturer_id = manufacturer.id
    paint_item.finish_id = finish.id
    paint_item.paint_medium_id = paint_medium.id
    paint_item.color = update_data["color"]
    paint_item.normalized_color=normalize_string(update_data["color"])
    paint_item.swatch = update_data["swatch"]
    paint_item.quantity_owned = update_data["quantity"]
    
    if update_data["quantity"] < 1:
        delete_paint(paint_id, db)
    else:
        add_com_ref(db, paint_item)
        
    return PaintOut(
        id=paint_id,
        manufacturer=update_data["manufacturer"],
        finish=update_data["finish"],
        medium=update_data["medium"],
        swatch=update_data["swatch"],
        quantity=update_data["quantity"],
        color=update_data["color"]
    )

@app.delete("/paints/{paint_id}", status_code=204)
def delete_paint(paint_id: int, db: Session = Depends(get_db)):
    statement = select(Paint).where(Paint.id == paint_id)
    paint: Paint | None = db.scalar(statement)
    if not paint:
        raise HTTPException(status_code=404, detail="Paint not found.")
    db.delete(paint)
    db.commit()
    return

@app.get("/manufacturers")
def get_manufacturers():
    return [e.value for e in ManufacturerEnum]

@app.get("/finishes")
def get_manufacturers():
    return [e.value for e in FinishEnum]

@app.get("/mediums")
def get_manufacturers():
    return [e.value for e in PaintMediumEnum]

@app.get("/upload_form", name="upload_form", response_class=HTMLResponse)
async def upload_form(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.post("/ocr", response_class=HTMLResponse)
async def run_ocr(
    request: Request,
    uploaded_files: list[UploadFile] = File(...)
) -> HTMLResponse:
    result_dicts: list[dict[str, str | None]] = []
    for file in uploaded_files:
        if not (
            file.filename.endswith('.png') or
            file.filename.endswith('.jpg') or
            file.filename.endswith('.jpeg') or
            file.filename.endswith('.webp')
        ):
            continue
        file_location = UPLOAD_DIR / file.filename
        with file_location.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result_dict: dict[str, str | None] = parse_image_as_string(
            file_location, OCR_DIR / file.filename
        )
        result_dict["filename"] = file.filename
        result_dict["uploaded_image"] = f"/static/uploads/{file.filename}"
        result_dict["ocr_image"] = f"/static/ocr/{file.filename}"
        result_dicts.append(result_dict)

    if not result_dicts:
        return templates.TemplateResponse("upload_form.html", {"request": request, "error": "No valid images uploaded."})
    
    # Store session data, need to add user/session management for this
    session_state["review_queue"] = result_dicts
    session_state["current_index"] = 0
    session_state["stored_paint"] = []
    
    return RedirectResponse("/review", status_code=303)

@app.get("/review")
async def review_image(request: Request) -> HTMLResponse:
    # replace with session management logic
    idx = session_state["current_index"]
    queue = session_state["review_queue"]

    if idx >= len(queue):  # No more images to review
        # Redirect to summary page
        return RedirectResponse("/summary", status_code=303)
    
    # Retrieve the current image data
    img_data: dict[str, str | None] = queue[idx]
    return templates.TemplateResponse("correct.html", {
        "request": request,
        **img_data
    })

@app.get("/summary", name="summary")
async def summary(request: Request) -> HTMLResponse:
    total_images: int = len(session_state["review_queue"])
    stored_paints: list[PaintDTO] = session_state["stored_paint"]
    color_list = [paint.color for paint in stored_paints] if stored_paints else ["No paints stored."]
    return templates.TemplateResponse(
        "summary.html", 
        {
            "request": request,
            "total_images": total_images,
            "stored_paints": len(stored_paints),
            "colors": color_list
        }
    )

@app.post("/correct", name="correction_form")
async def submit_data(
    request: Request,
    manufacturer: str = Form(...),
    color: str = Form(...),
    finish: str = Form(...),
    paint_medium: str = Form(...),
    action: str = Form(...),
    swatch: str | None = Form(None),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    idx = session_state["current_index"]
    queue = session_state["review_queue"]
    if idx < len(queue):
        if action == "submit":
            # Make and upsert a new Paint object
            paint: PaintDTO = PaintDTO(
                color=color,
                swatch=swatch,
                manufacturer=normalize_to_enum(manufacturer, ManufacturerEnum),
                finish=normalize_to_enum(finish, FinishEnum),
                paint_medium=normalize_to_enum(paint_medium, PaintMediumEnum),
            )
            upsert_paint(db, paint)
            session_state["stored_paint"].append(paint)
        session_state["current_index"] += 1
    return RedirectResponse("/review", status_code=303)
