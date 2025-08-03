"""Helper functions."""

from pathlib import Path

def get_font_path() -> Path:
    """Get the path to the Noto Sans JP Bold font."""
    # Assuming the font is located in a 'fonts' directory at the same level as this script
    fonts: Path = Path(__file__).parent.parent / "fonts" / "NotoSansJP-Bold.ttf"
    fonts = fonts.resolve()
    if not fonts.is_file():
        raise FileNotFoundError(f"Font file not found at: {fonts}")
    return fonts  