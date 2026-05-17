# Automatic 3D Motion Estimation for Rotomation

Convert an image sequence into a world-space 3D skeleton — ready for Autodesk Maya.
> **Note:** This repository contains reports and test materials alongside the tool.
> The pipeline you need is located at:
> ```
> COMP693_26S1_Project_Xinyu_Zhang_1163218/Deliverables/automatic_rotomation/
> ```
> All instructions below assume this folder as your working directory.

The pipeline chains three open-source tools into a single automated workflow:

- **YOLO26** — 2D pose detection and multi-person tracking
- **VideoPose3D** — lifting 2D keypoints to camera-space 3D joints
- **Custom global motion recovery** — placing the skeleton in world space using Maya camera extrinsics


## Requirements

- Python 3.9 or 3.10
- NVIDIA GPU with CUDA 11.x / 12.x recommended (CPU-only supported but slow)
- Autodesk Maya 2022–2025 (required only for camera export and final results visualization)


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/COMP693-Projects-26S1/COMP693_26S1_Project_Xinyu_Zhang_1163218.git
cd Deliverables/automatic_rotomation
```

### 2. Create a virtual environment

```bash
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 3. Install PyTorch

Install the version that matches your system from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/):

```bash
# NVIDIA GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# NVIDIA GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

## YOLO Model

The model `yolo26x-pose.pt` is downloaded automatically on first run and saved next to `detection_process.py`. No manual action is required.

For offline setups, download it manually and place it inside `automatic_rotomation/`:

```
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt
```


## Exporting Camera Data from Maya

The pipeline requires a `camera_data.json` file with per-frame camera intrinsics and extrinsics.

1. Open your Maya scene and load the shot.
2. Open the **Script Editor**.
3. Go to **File** -> **Source Script** -> find `export_camera_from_maya.py` under the repository and run it.
(Alternatively: Open the **Script Editor** -> switch to the **Python** tab. Paste the contents of `export_camera_from_maya.py` and run it.)
4. Select your camera node in the viewport or Outliner before running.
5. Fill in the start frame, end frame, and output path when prompted.

> The `start_frame` and `end_frame` in the exported JSON **must exactly match** the frame range you select during preprocessing.


## Running the Pipeline

```bash
python run.py
```

The pipeline will prompt you for the following inputs in order:

| Prompt | What to enter |
|--------|---------------|
| Project folder path | Absolute path to an output folder (created if it doesn't exist) |
| Image sequence folder | Absolute path to your JPEG frames |
| Start frame / End frame | Must match the camera JSON frame range |
| Select person track ID | Only shown when multiple people are detected |
| Character height (cm) | Actor's approximate height, e.g. `170` |
| Camera JSON path | Path to `camera_data.json` exported from Maya |

### Input requirements

- Frames must be JPEG (`.jpg` / `.jpeg`) — video files are not accepted
- Frame numbers must be consecutive with no gaps
- All frames must be the same resolution

Frames are resized to **640 × 384** during preprocessing.

## Project Structure

```
automatic_rotomation/
├── run.py
├── video_preprocess.py
├── detection_process.py
├── global_motion_recovery.py
├── display_plots.py
├── export_camera_from_maya.py
├── import_motion_into_maya.py
├── requirements.txt
├── yolo26x-pose.pt <- downloaded automatically on first run
└── VideoPose3D-main/
    ├── common/
    │   └── model.py
    └── checkpoint/
        └── pretrained_h36m_cpn.bin
```
## Output Files

| File | Description |
|------|-------------|
| `yolo_processed_video.mp4` | YOLO overlay video with detected skeletons |
| `keypoints_2d.npy` | Pixel-space 2D keypoints — shape (F, 17, 2) |
| `confidence.npy` | Per-joint confidence scores — shape (F, 17) |
| `pose_3d.mp4` | VideoPose3D side-by-side preview |
| `predicted_3d.npy` | Camera-space 3D joints — shape (F, 17, 3) |
| `predicted_3d_world.npy` | World-space 3D joints in cm — shape (F, 17, 3) |
| `predicted_3d_world.mp4` | Final animation with 2D overlay and 3D skeleton |

## Importing Motion into Maya
 
Once the pipeline has finished, use `import_motion_into_maya.py` to drive a joint skeleton in Maya directly from the `predicted_3d_world.npy` output.
 
1. Open your Maya scene.
2. Open the **Script Editor**.
3. Go to **File** -> **Source Script** -> find `import_motion_into_maya.py` under the repository and run it.
(Alternatively: Open the **Script Editor** -> switch to the **Python** tab. Paste the contents of `import_motion_into_maya.py` and run it.)
4. Two dialog boxes will appear:
   - **Pose Data** — enter the full path to `predicted_3d_world.npy` in your project folder
   - **Start Frame** — enter the frame you want the animation to begin on (defaults to the current timeline position)
### What it does
 
| Step | Description |
|------|-------------|
| Creates locators | One `LOC_<joint>` locator per joint, positioned and keyframed from the `.npy` data |
| Builds skeleton | A `predicted_<joint>` joint hierarchy matching the H36M layout |
| Applies constraints | Each joint is parent-constrained to its corresponding locator |
| Bakes animation | Constraints are baked to keyframes on the joints and removed |
 
### Joint hierarchy
 
```
Hip
├── RightHip → RightKnee → RightAnkle
├── LeftHip  → LeftKnee  → LeftAnkle
└── Spine → Thorax
              ├── Neck → Head
              ├── LeftShoulder  → LeftElbow  → LeftWrist
              └── RightShoulder → RightElbow → RightWrist
```
 
> Coordinates in `predicted_3d_world.npy` are in centimetres, matching Maya's default scene scale.
 
---

## Troubleshooting

**No images found in folder**
Only `.jpg` / `.jpeg` files are accepted. Convert PNG frames with:
```bash
mogrify -format jpg /path/to/frames/*.png
```

**Frame mismatch error**
The frame range in `camera_data.json` doesn't match the range entered during preprocessing. Re-export the camera JSON with the same frame numbers, or rerun preprocessing with the matching range.

**No people detected**
The subject may be too small after resizing (under ~80 × 80 px). Try a clip where the subject fills more of the frame, and check `yolo_processed_video.mp4` to see what YOLO detected.

**CUDA out of memory**
Process a shorter frame range, or force CPU mode:
```bash
CUDA_VISIBLE_DEVICES="" python run.py
```

**VideoPose3D import errors**
Ensure `VideoPose3D-main/` is placed directly inside `automatic_rotomation/` alongside `run.py`, and that `VideoPose3D-main/common/model.py` exists.

**Bone length warnings**
A `WARNING: Length not stable!` message means a bone varies more than 2 cm across frames, usually due to low joint confidence. Use a better-lit clip, or raise the confidence threshold in `global_motion_recovery.py` (the `is_valid_pair` function, default: `0.3`).

---

## Pre-Run Checklist

- [ ] Virtual environment is activated
- [ ] PyTorch installed for your system (GPU/CPU)
- [ ] `pip install -r requirements.txt` completed
- [ ] `VideoPose3D-main/` is placed inside `automatic_rotomation/` with `pretrained_h36m_cpn.bin` in its `checkpoint/` folder
- [ ] Input frames are JPEG with no gaps in frame numbers
- [ ] Camera JSON exported from Maya with matching start/end frames
- [ ] Sufficient disk space in the project folder (allow ~2 GB per shot)
- [ ] GPU drivers and CUDA toolkit match the installed PyTorch build