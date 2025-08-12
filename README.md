# Paint-Picker
Paint database with web-based frontend and some light OCR utilities sprinkled in.

<img width="1796" height="939" alt="image" src="https://github.com/user-attachments/assets/4537ec65-cf7d-4fbd-9dfc-8e74245cf5c1" />

<img width="605" height="1060" alt="image" src="https://github.com/user-attachments/assets/f47889cd-0585-49dd-a189-a0b62eae358d" />



# Initial Setup
To to setup your virtual environment, run:
```console
./create_venv.sh
```

You may need to activate the environment and install detectron2 manually with this command:
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

