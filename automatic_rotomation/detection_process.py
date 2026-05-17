import cv2
import os
import glob
import numpy as np
import sys
from requests import head
import torch
from ultralytics import YOLO
import urllib.request
from collections import Counter
from display_plots import run_videopose3d_preview

def run_yolo_pose_tracking(
    image_folder,
    project_folder,
    model_name="yolo26x-pose.pt",
    fps=24,
    selected_id=None
):
    frames = []

    # Load model
    model_path = os.path.join(os.path.dirname(__file__), model_name)
    if not os.path.exists(model_path):
        url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt"
        urllib.request.urlretrieve(url, model_path)

    model = YOLO(model_path)

    # Load images
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    if len(image_paths) == 0:
        raise FileNotFoundError("No images found in folder.")

    first_frame   = cv2.imread(image_paths[0])
    height, width = first_frame.shape[:2]

    # Output video
    output_video = os.path.join(project_folder, "yolo_processed_video.mp4")
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not out.isOpened():
        raise IOError("Failed to create output video writer")
    # Tracking storage
    all_track_ids = set()
    tracking_data = []

    print("Processing frames with YOLO tracking...")
    for img_path in image_paths:
        frame   = cv2.imread(img_path)
        # Detect and track people across frames, persist=True keeps consistent IDs
        results = model.track(frame, persist=True, verbose=False)
        result  = results[0]
        # Visualizes skeleton + bounding boxes
        overlay = result.plot() if (result.boxes is not None and len(result.boxes) > 0) else frame.copy()
        frames.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        out.write(overlay)

        frame_data = {}
        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            for i, track_id in enumerate(track_ids):
                all_track_ids.add(track_id)
                kp = result.keypoints[i].data.cpu().numpy() # (17, 3) → [x, y, confidence]
                if kp.ndim == 3:
                    kp = kp[0]
                # track_id : keypoints
                frame_data[track_id] = kp

        tracking_data.append(frame_data)

        cv2.imshow("YOLO Pose Tracking", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    out.release()
    cv2.destroyAllWindows()
    print(f"Processed {len(tracking_data)} frames.")

    # Select which person to continue tracking when multiple actors shown in the video
    if len(all_track_ids) == 0:
        raise RuntimeError("No people detected.")
    id_count = Counter()
    # Count how many frames each person appears
    for fd in tracking_data:
        for tid in fd:
            id_count[tid] += 1
    print("Person visibility (frame count):")
    for tid, count in id_count.items():
        print(f"ID {tid}: {count} frames")
    if selected_id is None:
        if len(all_track_ids) > 1:
            while True:
                user_input = input("Select person track ID to keep: ").strip()
                if not user_input.isdigit():
                    print("Error: Please enter a valid numeric ID.")
                    continue
                selected_id = int(user_input)
                if selected_id not in all_track_ids:
                    print(f"Error: ID {selected_id} not found. Available IDs: {list(all_track_ids)}")
                    continue
                break
        else:
            # If only one person is detected
            selected_id = list(all_track_ids)[0]
            print(f"Only one person detected, using ID: {selected_id}")
    print(f"Using track ID: {selected_id}")


    # COCO → H36M  (single loop, three outputs)
    # keypoints_px:   pixel coords          for reprojection optimisation
    # confidence_arr: per-joint conf scores for reprojection optimisation
    # keypoints_norm: normalised [-1,1]     for VideoPose3D

    num_frames     = len(tracking_data)
    keypoints_px   = np.zeros((num_frames, 17, 2), dtype=np.float32)
    confidence_arr = np.zeros((num_frames, 17),    dtype=np.float32)
    keypoints_norm = np.zeros((num_frames, 17, 2), dtype=np.float32)
    
    # From raw YOLO joints to reconstructed human skeleton
    def coco_to_h36m(kp):
        """
        kp: (17, 3) COCO keypoints — x, y, conf
        Returns px (17,2), conf (17,)
        """
        # Virtual joints (position)
        pelvis = (kp[11, :2] + kp[12, :2]) / 2
        thorax   = (kp[5,  :2] + kp[6,  :2]) / 2
        spine  = (kp[5,  :2] + kp[6,  :2] + kp[11, :2] + kp[12, :2]) / 4
        head   = kp[0,  :2]
        neck     = (thorax + head ) / 2

        # Virtual joints (confidence)
        pelvis_c = (kp[11, 2] + kp[12, 2]) / 2
        thorax_c   = (kp[5,  2] + kp[6,  2]) / 2
        spine_c  = (kp[5,  2] + kp[6,  2] + kp[11, 2] + kp[12, 2]) / 4
        head_c   = kp[0,  2] 
        neck_c    = (thorax_c + head_c) / 2

        px = np.array([
            pelvis,       # 0  Hip
            kp[12, :2],   # 1  RHip
            kp[14, :2],   # 2  RKnee
            kp[16, :2],   # 3  RAnkle
            kp[11, :2],   # 4  LHip
            kp[13, :2],   # 5  LKnee
            kp[15, :2],   # 6  LAnkle
            spine,        # 7  Spine
            thorax,       # 8  Thorax
            neck,         # 9  Neck
            head,         # 10 Head
            kp[5,  :2],   # 11 LShoulder
            kp[7,  :2],   # 12 LElbow
            kp[9,  :2],   # 13 LWrist
            kp[6,  :2],   # 14 RShoulder
            kp[8,  :2],   # 15 RElbow
            kp[10, :2],   # 16 RWrist
        ], dtype=np.float32)

        conf = np.array([
            pelvis_c,     # 0
            kp[12, 2],    # 1
            kp[14, 2],    # 2
            kp[16, 2],    # 3
            kp[11, 2],    # 4
            kp[13, 2],    # 5
            kp[15, 2],    # 6
            spine_c,      # 7
            thorax_c,     # 8
            neck_c,       # 9
            head_c,       # 10
            kp[5,  2],    # 11
            kp[7,  2],    # 12
            kp[9,  2],    # 13
            kp[6,  2],    # 14
            kp[8,  2],    # 15
            kp[10, 2],    # 16
        ], dtype=np.float32)

        return px, conf

    for f, frame_data in enumerate(tracking_data):
        if selected_id not in frame_data:
            if f > 0:
                keypoints_px[f]   = keypoints_px[f - 1]
                confidence_arr[f] = confidence_arr[f - 1] * 0.5
                keypoints_norm[f] = keypoints_norm[f - 1]
                print(f"Warning: Frame {f}: ID {selected_id} not detected, reusing previous frame")
            else:
                print(f"Warning: Frame {f}: ID {selected_id} not detected on first frame, skipping")
            continue
        # Get keypoints
        kp = frame_data[selected_id]
        if kp.ndim == 3:
            kp = kp[0]
        if kp.shape[0] != 17:
            continue
        if np.all(kp[:, :2] == 0):
            continue
        # Convert to skeleton
        px, conf = coco_to_h36m(kp)
        # Store pixel coordinates
        keypoints_px[f]   = px
        confidence_arr[f] = conf

        # Normalise pixel to [-1, 1] for VideoPose3D
        keypoints_norm[f, :, 0] = (px[:, 0] / width)  * 2 - 1
        keypoints_norm[f, :, 1] = (px[:, 1] / height) * 2 - 1
        # # flip X axis
        # keypoints_norm[f, :, 0] *= -1

    # Save
    kp_path   = os.path.join(project_folder, "keypoints_2d.npy")
    conf_path = os.path.join(project_folder, "confidence.npy")
    np.save(kp_path,   keypoints_px)
    np.save(conf_path, confidence_arr)

    print(f"\nSaved outputs:")
    print(f"video:          {output_video}")
    print(f"keypoints_2d:   {kp_path}    {keypoints_px.shape}")
    print(f"confidence:     {conf_path}  {confidence_arr.shape}")

    return keypoints_norm, frames, keypoints_px, confidence_arr  # keypoints_norm → VideoPose3D

def run_videopose3d(
    keypoints_2d,          # numpy array (frames, 17, 2)
    frames,                # list of RGB frames (for visualization)
    project_folder,
    videopose3d_path,
    checkpoint_path
):
    """
    Run VideoPose3D to predict 3D pose and render video.
    Args:
        keypoints_2d: numpy array of shape (frames, 17, 2) with normalized keypoints.
        frames: list of RGB frames (for visualization).
        project_folder: Path to save outputs.
        videopose3d_path: Path to VideoPose3D code.
        checkpoint_path: Path to VideoPose3D pretrained model checkpoint.
    Returns:
        output_3d: Path to saved 3D video.
    """
    
    # Setup VideoPose3D
    sys.path.append(videopose3d_path)
    from common.model import TemporalModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create model
    model_3d = TemporalModel(
        num_joints_in=17,
        num_joints_out=17,
        in_features=2,
        filter_widths=[3,3,3,3,3],
        causal=False
    ).to(device)
    # Load pretrained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_3d.load_state_dict(checkpoint["model_pos"])
    model_3d.eval()

    # Pad temporal window, add fake frames at start/end
    pad = 121
    poses_2d = np.pad(keypoints_2d, ((pad,pad),(0,0),(0,0)), mode="edge")
    # Convert to PyTorch format
    inputs_2d = torch.tensor(poses_2d, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict 3D
    print("Running VideoPose3D...")
    with torch.no_grad():
        predicted_3d = model_3d(inputs_2d).squeeze(0).cpu().numpy()
    
    # Flip Z axis (coordinate system fix)
    predicted_3d[:,:,2] *= -1
    # flip X axis to correct left/right
    predicted_3d[:,:,0] *= -1

    # Save 3D joint positions to .npy
    npy_path = os.path.join(project_folder, "predicted_3d.npy")
    np.save(npy_path, predicted_3d)
    print("Saved 3D joint positions to:", npy_path)

    # Render
    print("Rendering 3D video...")
    preview_path = os.path.join(project_folder, "pose_3d.mp4")
    run_videopose3d_preview(frames, predicted_3d, preview_path)

    return predicted_3d
