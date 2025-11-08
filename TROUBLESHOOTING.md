# Troubleshooting: Python version compatibility (Windows)

Your current system Python is 3.13 (`C:/Python313/python.exe`). Some ML packages in this project (notably `mediapipe` and `tensorflow`) do not yet provide wheels for Python 3.13 on Windows. This causes errors like:

```
ERROR: Could not find a version that satisfies the requirement mediapipe>=0.10.0 (from versions: none)
ERROR: No matching distribution found for mediapipe>=0.10.0
```

## Recommended fix: Use Python 3.11 virtual environment

MediaPipe and TensorFlow work reliably on Python 3.11 today. Create a 3.11 venv and install the 3.11-locked requirements.

### Steps (PowerShell)

1) Install Python 3.11 (64-bit) from python.org
- Download: https://www.python.org/downloads/release/python-3110/
- Ensure you tick "Add Python to PATH" during installation.

2) Verify `py` launcher sees Python 3.11

```powershell
py -0p
```
You should see a line like `-V:3.11  C:\\Path\\to\\python311\\python.exe`.

3) Create and activate a venv with 3.11

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

4) Install the 3.11-compatible requirements

```powershell
pip install -r requirements-311.txt
```

If you prefer to keep using `requirements.txt`, you can try it as well on Python 3.11:

```powershell
pip install -r requirements.txt
```

## Why not Python 3.13?

- `mediapipe` and `tensorflow` often lag behind the newest Python releases on Windows. As of late 2025, official wheels for 3.13 may not be published for all combinations yet.
- Using 3.11 avoids build-from-source scenarios and ensures smooth installation.

## Optional: GPU builds

If you have an NVIDIA GPU and want CUDA acceleration:
- Install PyTorch per your CUDA version: https://pytorch.org/get-started/locally/
- Then pin matching `torch`/`torchvision` versions and reinstall `timm` if needed.

## Still stuck?

- Share the output of these commands:
  - `py -0p`
  - `python --version`
  - `pip debug --verbose`
- And the full error log from `pip install`.
