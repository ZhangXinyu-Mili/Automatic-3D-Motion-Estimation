# Test Materials

Test shots are provided so you can verify the pipeline is working correctly before using your own footage. They are located at:

```
Automatic-3D-Motion-Estimation/test_shots/
```


## Available Shots
- Agent Runs from Explosion
- Man running up stairs
- Man Wakes in Abandoned Building
- Post-Apocalyptic Drone Attack


## Contents of Each Shot Folder

Each shot folder contains:

- **`.mp4` video** — reference video of the shot
- **`.mb` Maya file** — Maya scene with the camera, image plane, and basic set geometry already set up
- **`camera_data.json`** — exported camera extrinsics ready to use directly with the pipeline
- **`pre_computed_output/`** — pre-computed reference outputs you can use to skip straight to the Maya import step, or compare against your own pipeline results

| File | Description |
|------|-------------|
| `predicted_3d_world.npy` | World-space 3D joints in cm — the main output used for Maya import |
| `predicted_3d_world.mp4` | Final animation with 2D overlay and 3D skeleton |
| `pose_3d.mp4` | VideoPose3D side-by-side preview |
| `yolo_processed_video.mp4` | YOLO overlay video with detected skeletons |
| `predicted_3d.npy` | Camera-space 3D joints |
| `keypoints_2d.npy` | Pixel-space 2D keypoints |
| `confidence.npy` | Per-joint confidence scores |
 
> **Image sequences are not included in the repository** due to GitHub file size limits.
> They are available on Google Drive — see the link below.

## Image Sequences (Google Drive)

The JPEG image sequences for all four shots are hosted on Google Drive:

**[Download image sequences → Google Drive](https://drive.google.com/file/d/1hjqyEsrNG4T2R3T0s4sbb4ggnRRb5klq/view?usp=drive_link)**

Download the folder for the shot you want to test and note the path — you will be prompted for it when you run the pipeline.

## Folder Structure

```
test_shots/
├── agent_runs_from_explosion/
│   ├── agent_runs_from_explosion.mp4
│   ├── agent_runs_from_explosion.mb
│   ├── camera_data.json
│   └── pre_computed_output/
│       ├── predicted_3d_world.npy
│       ├── predicted_3d_world.mp4
│       ├── pose_3d.mp4
│       ├── yolo_processed_video.mp4
│       ├── predicted_3d.npy
│       ├── keypoints_2d.npy
│       └── confidence.npy
├── man_running_up_stairs/
│   ├── man_running_up_stairs.mp4
│   ├── man_running_up_stairs.mb
│   ├── camera_data.json
│   └── pre_computed_output/
│       ├── predicted_3d_world.npy
│       ├── predicted_3d_world.mp4
│       ├── pose_3d.mp4
│       ├── yolo_processed_video.mp4
│       ├── predicted_3d.npy
│       ├── keypoints_2d.npy
│       └── confidence.npy
├── man_wakes_in_abandoned_building/
│   ├── man_wakes_in_abandoned_building.mp4
│   ├── man_wakes_in_abandoned_building.mb
│   ├── camera_data.json
│   └── pre_computed_output/
│       ├── predicted_3d_world.npy
│       ├── predicted_3d_world.mp4
│       ├── pose_3d.mp4
│       ├── yolo_processed_video.mp4
│       ├── predicted_3d.npy
│       ├── keypoints_2d.npy
│       └── confidence.npy
└── post_apocalyptic_drone_attack/
    ├── post_apocalyptic_drone_attack.mp4
    ├── post_apocalyptic_drone_attack.mb
    ├── camera_data.json
    └── pre_computed_output/
        ├── predicted_3d_world.npy
        ├── predicted_3d_world.mp4
        ├── pose_3d.mp4
        ├── yolo_processed_video.mp4
        ├── predicted_3d.npy
        ├── keypoints_2d.npy
        └── confidence.npy
```

> Image sequences are not shown above — download them from Google Drive and keep them separate from this folder.