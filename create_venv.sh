
VENV_NAME=".venv"
EDITABLE="true"

echo "Using '$VENV_NAME' for virtual environment."

uv venv \
    --relocatable \
    --link-mode "copy" \
    $VENV_NAME

UV_EXTRAS=""
if [ "$EDITABLE" == "false" ]; then
    UV_EXTRAS="--no-editable"
fi

uv sync --frozen --active $UV_EXTRAS --all-packages