from pathlib import Path
import shutil
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, Engine
from src.models import Base, Paint, PaintDTO
from src.paint_parser import parse_image_as_string
from src.update_db import upsert_paint
from src.database_helpers import normalize_to_enum
from src.base_classes import FinishEnum, PaintMediumEnum, ManufacturerEnum


DATABASE_URL: str = "sqlite:///./paintdb.sqlite3"
engine: Engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(engine)
UPLOAD_DIR: Path = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OCR_DIR: Path = Path("static/ocr")
OCR_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
templates= Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.post("/ocr", response_class=HTMLResponse)
async def run_ocr(
    request: Request,
    file: UploadFile = File(...)
) -> HTMLResponse:
    if not (
        file.filename.endswith('.png') or
        file.filename.endswith('.jpg') or
        file.filename.endswith('.jpeg') or
        file.filename.endswith('.webp')
    ):
        return templates.TemplateResponse(
            "upload_form.html",
            {"request": request, "error": "Only png/jpeg/webp files are allowed."}
        )
    file_location = UPLOAD_DIR / file.filename
    with file_location.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result_dict: dict[str, str | None] = parse_image_as_string(
        file_location, OCR_DIR / file.filename
    )
    return templates.TemplateResponse("correct.html", {
        "request": request,
        "uploaded_image": f"/static/uploads/{file.filename}",
        "ocr_image": f"/static/ocr/{file.filename}",
        **result_dict
    })

@app.post("/submit")
async def submit_data(
    manufacturer: str = Form(...),
    color: str = Form(...),
    finish: str = Form(...),
    paint_medium: str = Form(...)
) -> RedirectResponse:
    with Session(engine) as session:
        # Make and upsert a new Paint object
        paint: PaintDTO = PaintDTO(
            color=color,
            manufacturer=normalize_to_enum(manufacturer, ManufacturerEnum),
            finish=normalize_to_enum(finish, FinishEnum),
            paint_medium=normalize_to_enum(paint_medium, PaintMediumEnum),
        )
        upsert_paint(session, paint)
    return RedirectResponse("/", status_code=303)
