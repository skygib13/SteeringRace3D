# SteeringRace3D — Computer Vision Racing Game

A Python 3D racing game controlled by a virtual steering wheel: rotate your two hands in front of a webcam to steer. Built with **Ursina**, **OpenCV**, and **MediaPipe**.

## Requirements

- Python 3.10 (64-bit recommended)
- Webcam (optional: keyboard control is available)

## Setup (Windows / PowerShell)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main_3d.py
```

Alternatively, double-click `start_windows.bat` after installing dependencies.

## Controls

| Input | Action |
|---|---|
| Two hands rotated | Steer |
| C | Calibrate virtual wheel |
| A / D or arrows | Keyboard steering |
| M | Toggle camera/keyboard |
| W / S | Accelerate / brake |
| Enter | Start race |
| P | Pause/resume |
| R | Restart |
| Esc | Exit |

## Main files

- `main_3d.py`: game logic, road, vehicles, HUD
- `camera_steering.py`: webcam and hand landmarks
- `steering_math.py`: steering angle calculations
- `test_3d.py`: smoke test
- `requirements.txt`: pinned Python dependencies

See [README_KZ.md](README_KZ.md) for the detailed guide in Kazakh.

## Notes

This is a prototype, and hands are recognized through the webcam rather than a physical wheel. The project is intended for local Python execution, not GitHub Pages.
