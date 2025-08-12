# Paint-Picker
Paint database with web-based frontend and some light OCR utilities sprinkled in.


# Initial Setup
To to setup your virtual environment, run:
```console
./create_venv.sh
```

You may need to acivate it and install detectron2 manually with this command:
```console
pip3 install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation 
```

I have had some trouble using uv to make Paddle and Pydorch play nice resolving dependencies.
I uninstall it, and install via pip to get a CUDA version, which works with Paddle.
```console
pip3 uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

For convenience `create_venv.sh` should do all of this automatically.

# Running
Launch the server with:
```console
uvicorn src.main:app
```

