
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

uv sync --frozen --active $UV_EXTRAS --all-packages --index-strategy unsafe-best-match
source .venv/bin/activate
python -m ensurepip
pip3 install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
pip3 uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128