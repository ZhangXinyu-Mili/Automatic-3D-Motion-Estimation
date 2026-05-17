# Automatic 3D Motion Estimation
This project develops a lightweight automatic rotomation tool that generates approximate 3D character motion from monocular video footage. In the VFX industry, rotomation is traditionally a slow and repetitive process requiring artists to manually recreate movement frame-by-frame. The tool aims to reduce this workload by providing animators with a rough starting point for previsualisation and early-stage animation workflows. 

The tool combines video preprocessing, 2D pose detection, 3D pose estimation, global motion recovery, and motion refinement. OpenCV is used for video preprocessing, while pose estimation models such as YOLO and VideoPose3D reconstruct 3D skeletal motion from detected 2D keypoints. Camera-aware optimisation is applied to recover world-space translation and rotation, followed by temporal smoothing to reduce jitter and improve stability. 

The final reconstructed motion is exported in .npy format and imported into Autodesk Maya for visualisation and retargeting. Testing demonstrated usable motion reconstruction, improved workflow efficiency, and reduced manual animation effort. 

# Pipeline Architecture and Design

The tool is designed as a sequential motion reconstruction pipeline where each component processes data and passes its output to the next stage.

The architecture combines:

- computer vision
- machine learning
- optimisation
- animation export workflows

to convert monocular video footage into approximate 3D motion suitable for rotomation and previsualisation.

Development was conducted in Python using standard scientific computing libraries including:

- NumPy
- SciPy
- OpenCV

alongside deep learning frameworks:

- PyTorch
- Ultralytics

## Pipeline Stages

The pipeline consists of **4 sequential processing stages**, implemented across **5 Python modules**.

A Maya-side prerequisite script must be run before the main pipeline begins in order to export camera data.

| Stage | Module | Input | Output |
|---|---|---|---|
| Pre | `export_camera_from_maya.py` *(runs inside Maya)* | Maya scene + camera node | `camera_data.json` |
| 1 | `video_preprocess.py` | Raw JPEG sequence | Resized `640×384` frames |
| 2–3 | `detection_process.py` | Preprocessed frames | 2D keypoints, confidence maps, camera-space 3D joints, preview videos |
| 4 | `global_motion_recovery.py` | 3D joints + camera JSON | World-space 3D joints |
| 5 | `import_motion_into_maya.py` *(runs inside Maya)* | `predicted_3d_world.npy` | Baked Maya joint animation |


## Pipeline Overview

```text
Maya Camera Export
        ↓
Video Preprocessing
        ↓
YOLO Pose Detection
        ↓
VideoPose3D Reconstruction
        ↓
Global Motion Recovery
        ↓
IK Refinement
        ↓
World-Space Motion Export
        ↓
Maya Import and Baking
```

# Data Flow Between Modules

## 1. Maya Camera Export

`export_camera_from_maya.py` runs independently inside Maya before the pipeline begins.

The script:

- reads the camera node world matrix at every frame
- extracts intrinsic and extrinsic camera parameters
- exports them into `camera_data.json`

This JSON file acts as the only coupling point between Maya and the Python pipeline.

---

## 2. Video Preprocessing

`video_preprocess.py`:

- reads raw JPEG image sequences
- validates frame continuity
- checks resolution consistency
- resizes frames to `640 × 384`
- writes processed frames into a `preprocessed/` folder

The module returns:

- processed image folder path
- frame range information

back to `run.py`.

---

## 3. Detection and 3D Reconstruction

`detection_process.py` receives the preprocessed frames from `run.py`.

The module performs:

### YOLO Pose Detection

- tracks the subject
- extracts 2D keypoints in COCO format

### Skeleton Conversion

- converts COCO joints into H36M format using `coco_to_h36m()`

### VideoPose3D Inference

- normalises 2D keypoints
- reconstructs camera-space 3D joints

### Output Generation

The module saves:

- `keypoints_2d.npy`
- `confidence.npy`

and returns:

- normalised 2D keypoints
- RGB frames
- pixel-space keypoints
- confidence arrays
- camera-space 3D joints

directly to `run.py`.

---

## 4. Global Motion Recovery

`global_motion_recovery.py` receives:

- camera-space 3D joints
- pixel-space 2D keypoints
- confidence arrays
- camera intrinsics
- camera extrinsics

from `run.py`.

The module:

- reconstructs world-space motion
- refines joint positions
- applies IK optimisation
- outputs world-space skeleton motion

Results are:

- returned to `run.py`
- saved as `predicted_3d_world.npy`

---

## 5. Maya Motion Import

`import_motion_into_maya.py` runs inside Maya after pipeline completion.

The script:

- loads `predicted_3d_world.npy`
- creates locators and joints
- keyframes locator motion
- applies constraints
- bakes animation
- removes constraints

This produces a clean baked animation skeleton inside Maya.

---

# Algorithm Logic

The core technical contribution of this project is the `global_motion_recovery.py` module.

VideoPose3D produces 3D joints in **camera space**, meaning the skeleton remains attached to the camera coordinate system.

For VFX workflows involving moving cameras, this output is not directly usable inside a Maya scene.

The global motion recovery system converts camera-space motion into stable world-space motion using a six-stage reconstruction pipeline.

---

# 1. Root-Relative Pose Extraction

The raw VideoPose3D output contains absolute camera-space coordinates for every joint.

The pelvis joint (joint index `0`) is subtracted from all joints to generate a root-relative skeleton.

This isolates:

- body proportions
- limb motion
- local pose structure

from global movement.

---

# 2. Bone Length Enforcement

VideoPose3D predictions often contain inconsistent bone lengths across frames.

To correct this, each bone is scaled toward a canonical target length derived from anthropometric body proportions.

The correction formula is:

```text
corrected_vec = vec × (1 + ((target_len / current_len) − 1) × blend)
```

Where:

- `vec` = current bone vector
- `target_len` = canonical bone length
- `current_len` = predicted bone length
- `blend` = correction strength (`1.0`)

Canonical bone lengths are estimated as fixed percentages of actor height.

Examples include:

- thigh = 24.6% of height
- upper arm = 15.0% of height

This improves anatomical consistency and reduces visual jitter.

---

# 3. Depth Estimation from Canonical Bone Lengths

Recovering absolute depth from monocular footage is inherently ambiguous.

The root joint depth is estimated using the perspective projection relationship:

```text
Z ≈ (f × d3D) / d2D
```

Where:

- `f` = focal length
- `d3D` = canonical 3D bone length
- `d2D` = observed 2D pixel bone length

The method assumes that objects appear smaller as they move further from the camera.

---

## Bone Priority System

To improve robustness:

1. full-body proxy bones are evaluated first
2. upper-body bones second
3. lower-body bones third
4. arms last

A bone is only accepted if:

- YOLO confidence > `0.4`
- pixel length > `5 px`

This avoids unstable depth estimates from noisy detections.

---

## Temporal Smoothing

Raw depth estimates are smoothed using a:

- 15-frame uniform temporal filter

This reduces frame-to-frame jitter.

---

# 4. Root Back-Projection to World Space

The root joint is back-projected into 3D camera space using:

```text
Xc = ((u − cx) / fx) × Z
Yc = ((v − cy) / fy) × Z
Zc = Z
```

Where:

- `(u, v)` = 2D pixel coordinates
- `(cx, cy)` = camera principal point
- `(fx, fy)` = focal lengths

The resulting camera-space root position is converted into world space using the inverse camera extrinsic transform:

```text
P_world = R_cam^T × (P_cam − t_cam)
```

Where:

- `R_cam` = camera rotation matrix
- `t_cam` = camera translation vector

This anchors the character into scene space.

---

# 5. Pose Assembly in World Space

Root-relative joint offsets remain expressed in camera space.

These offsets are rotated into world space using:

```text
P_world[j] = R_cam^T × P_local[j] + P_root_world
```

This reconstructs the complete world-space skeleton pose for every frame.

---

# 6. Reprojection-Based IK Refinement

The final stage performs partial inverse kinematics optimisation using SciPy's `L-BFGS-B` solver.

A weighted multi-objective loss function is minimised over a subset of 14 joints.

---

## Optimisation Loss Terms

| Loss Term | Weight | Purpose |
|---|---|---|
| Reprojection error | `3.0` | Minimises projected 2D joint error |
| Bone length error | `10.0` | Preserves anatomical proportions |
| Joint angle penalty | `0.2` | Prevents unrealistic hyperextension |
| Shoulder constraint | `3.0` | Enforces shoulder symmetry |
| Hip constraint | `3.0` | Preserves hip alignment |
| Neck/head constraint | `3.0` | Maintains head-neck-thorax ordering |

---

## Optimisation Strategy

The optimiser runs for a maximum of:

```text
15 iterations per frame
```

The high weighting on bone length preservation ensures anatomical plausibility takes precedence over noisy 2D detections.

The refined output becomes:

```text
predicted_3d_world.npy
```

which represents the final world-space motion reconstruction result.