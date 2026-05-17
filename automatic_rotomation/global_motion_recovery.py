import os
import cv2
import glob
import json
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.optimize import minimize
from scipy.ndimage import uniform_filter1d

# JOINT INDICES (H36M)
# 0:Hip  1:RHip  2:RKnee  3:RAnkle
# 4:LHip 5:LKnee 6:LAnkle
# 7:Spine 8:Thorax 9:Neck 10:Head
# 11:LShoulder 12:LElbow 13:LWrist
# 14:RShoulder 15:RElbow 16:RWrist
SKELETON = [(0,1),(1,2),(2,3),
        (0,4),(4,5),(5,6),
        (0,7),(7,8),(8,9),
        (9,10),(8,11),(11,12),
        (12,13),(8,14),(14,15),(15,16),]

OPTIMIZE_JOINTS = [2,3,5,6,7,8,9,10,11,12,13,14,15,16]

BONE_GROUPS = [
    # Full body proxy (long + stable)
    [(0,8)],   # hip→spine (long bone, stable)

    # Upper body
    [(8,9), (7,8)],   # thorax→neck, spine→thorax

    # Lower body
    [(1,2), (2,3), (4,5), (5,6)],  # legs

    # Arms (critical for hand width matching)
    [(11,12), (12,13)],   # LShoulder→LElbow→LWrist
    [(14,15), (15,16)],  # RShoulder→RElbow→RWrist
    [(11,14)],           # shoulder span (full arm width)
]

JOINT_ANGLE_LIMITS = {
    (1,2,3): (0, 160),   # knees
    (4,5,6): (0, 160),
    (11,12,13): (0, 160),  # elbows
    (14,15,16): (0, 160),
    # (0,7,8): (-60, 60),   # spine
    # (7,8,9): (-60, 60),
    # (8,9,10): (-60, 60),  # neck
}

def load_frames(image_folder, pose3d, camera_json, start_frame=None, end_frame=None):
    """
    Load RGB frames from disk and assemble all data needed for the solve.
 
    Accepts pose3d, keypoints_2d, and confidence as in-memory arrays (passed
    directly from the detection step) rather than file paths.
 
    Args:
        image_folder (str): Path to the folder containing preprocessed images.
        pose3d (np.ndarray): Shape (F, J, 3), raw 3D joint positions in metres.
        keypoints_2d (np.ndarray): Shape (F, J, 2), pixel-space 2D keypoints.
        confidence (np.ndarray or None): Shape (F, J), per-joint confidence scores.
        camera_json (str): Path to the camera data JSON file.
 
    Returns:
        frames (list of np.ndarray): List of RGB frames.
        preprocess_w (int): Width of the preprocessed frames.
        preprocess_h (int): Height of the preprocessed frames.
        poses_3d (np.ndarray): Shape (F, J, 3), 3D joint positions in cm with
            coordinate convention fixed.
        keypoints_2d (np.ndarray): Shape (F, J, 2), 2D joint positions.
        confidence (np.ndarray or None): Shape (F, J), or None if unavailable.
        K (np.ndarray): Camera intrinsic matrix scaled to preprocess resolution.
        R_frames (np.ndarray): Shape (F, 3, 3), per-frame camera rotations.
        t_frames (np.ndarray): Shape (F, 3), per-frame camera translations.
        F (int): Number of frames.
    """
    # define frames as list of RGB images (for visualization)
    frames = []
    # store detected size
    preprocess_h, preprocess_w = None, None  
    paths  = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    if len(paths) == 0:
        raise FileNotFoundError("No images found in folder.")
    
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            raise ValueError(f"Failed to load image: {p}")
        h, w = img.shape[:2]
        # store detected size from first frame
        if i == 0:
            preprocess_h, preprocess_w = h, w
            print(f"Detected image size: {preprocess_w}x{preprocess_h}")
        else:
            # Ensure all frames match
            if h != preprocess_h or w != preprocess_w:
                raise ValueError(
                    f"Inconsistent image size at {p}: got {w}x{h}, expected {preprocess_w}x{preprocess_h}"
                )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    print(f"Loaded {len(frames)} frames from {image_folder}")
    
    # Load 3D pose, 2D keypoints, confidence, and camera data

    # multiply pose3d_npy by 100 → converting from meters → cm (matches Maya scene scale)
    poses_3d     = pose3d* 100.0
    # Fix coordinate convention
    poses_3d[:, :, 0] *= -1   # mirror X: fix left/right swap
    # poses_3d[:, :, 1] *= -1   # flip Y: image-down → world-up  
    poses_3d[:, :, 2] *= -1   # flip Z: toward-camera → away-from-camera

    # Load camera data
    with open(camera_json) as f:
        cam_data = json.load(f)
    # Get camera start and end frame
    cam_start = cam_data["start_frame"]
    cam_end = cam_data["end_frame"]
    # Validate if the frame range match with the preprocessed image frame range
    if cam_start != start_frame or cam_end != end_frame:
        raise ValueError(
            f"Error: Frame mismatch!\n"
            f"Camera JSON: {cam_start}-{cam_end}\n"
            f"Preprocess:   {start_frame}-{end_frame}"
        )
    # Get camera intrinsics
    K_original = np.array(cam_data["K"])
    orig_w     = cam_data["render_width"]
    orig_h     = cam_data["render_height"]
    K = K_original.copy()
    # fx, fy, cx, cy need to be scaled according to the actual size of the preprocessed frames
    K[0, 0] *= preprocess_w / orig_w
    K[1, 1] *= preprocess_h / orig_h
    K[0, 2] *= preprocess_w / orig_w
    K[1, 2] *= preprocess_h / orig_h
    # Get camera extrinsics
    R_frames = np.array([fr["R"] for fr in cam_data["frames"]])
    t_frames = np.array([fr["t"] for fr in cam_data["frames"]])
    # Validate all inputs have the same number of frames as 3D pose
    F = poses_3d.shape[0]
    # assert keypoints_2d.shape[0] == F, f"keypoints_2d mismatch: {keypoints_2d.shape[0]} vs {F}"
    assert R_frames.shape[0]     == F, f"camera mismatch: {R_frames.shape[0]} vs {F}"
    assert len(frames)           == F, f"image mismatch: {len(frames)} vs {F}"
    print(f"All inputs validated — {F} frames")

    return frames, preprocess_w, preprocess_h, poses_3d, K, R_frames, t_frames, F


def project_points(points_3d, K, R_cam, t_cam):
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d (np.ndarray): Array of shape (J, 3) with 3D joint positions.
        K (np.ndarray): Camera intrinsic matrix.
        R_cam (np.ndarray): Camera rotation matrix.
        t_cam (np.ndarray): Camera translation vector.

    Returns:
        projected_points (np.ndarray): Array of shape (J, 2) with 2D pixel coordinates.
        valid (np.ndarray): Boolean array of shape (J,) indicating if each point is in front of the camera.
    """

    # Take every (X,Y,Z) point, rotate in 3D, shift in 3D, then project to 2D
    pts_cam = (R_cam @ points_3d.T).T + t_cam
    # Only points with Z > 0 are in front of the camera and can be projected
    # Z <= 0 means behind camera means invalid projection
    valid   = pts_cam[:, 2] > 0
    # normalized screen position in camera space (divide by Z depth)
    x = np.where(valid, pts_cam[:, 0] / pts_cam[:, 2], 0)
    y = np.where(valid, pts_cam[:, 1] / pts_cam[:, 2], 0)
    # convert to pixels using intrinsics
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    # combines u and v into a single 2D coordinate per joint
    return np.stack([u, v], axis=-1), valid

def get_canonical_bone_lengths(height_cm):
    """
    Get canonical bone lengths based on a standard human skeleton scaled to the actor's height.
    
    Args:
        height_cm (float): The height of the actor in centimeters.

    Returns:
        dict: A dictionary mapping (parent_joint, child_joint) to bone length in cm.
    """
    
    H = height_cm
    lengths = {}
    # -Spine-
    lengths[(0,7)] = 0.14 * H     # Hip to spine 
    lengths[(7,8)] = 0.13 * H     # Spine to thorax 
    lengths[(8,9)] = 0.05 * H     # Thorax to neck 
    lengths[(9,10)] = 0.05 * H    # Neck to head

    # -Legs (Standard)-
    lengths[(0,1)] = 0.08 * H     # Hip to RHip
    lengths[(1,2)] = 0.246 * H   # RHip to RKnee
    lengths[(2,3)] = 0.247 * H  # RKnee to RAnkle
    lengths[(0,4)] = 0.08 * H    # Hip to LHip
    lengths[(4,5)] = 0.246 * H   # LHip to LKnee
    lengths[(5,6)] = 0.247 * H   # LKnee to LAnkle

    # -Shoulders-
    lengths[(8,11)] = 0.13 * H    # Thorax center to L Shoulder
    lengths[(8,14)] = 0.13 * H    # Thorax center to R Shoulder
    lengths[(11,14)] = 0.26 * H   # TOTAL Shoulder span (The Depth Driver)

    # -Arms-
    lengths[(11,12)] = 0.150 * H  # LShoulder to LElbow
    lengths[(12,13)] = 0.146 * H  # LElbow to LWrist
    lengths[(14,15)] = 0.150 * H  # RShoulder to RElbow
    lengths[(15,16)] = 0.146 * H  # RElbow to RWrist

    return lengths

def enforce_bone_lengths(pose, skeleton, canonical_bone_lengths, blend=0.5):
    """
    Enforce bone lengths by adjusting joint positions.

    Args:
        pose (np.ndarray): Array of shape (J, 3) with 3D joint positions.
        skeleton (list of tuples): List of (parent, child) joint index pairs.
        canonical_bone_lengths (dict): Dictionary mapping (parent, child) to target bone length.
        blend (float): Blending factor between current and target length (0 = no correction, 1 = full correction).

    Returns:
        np.ndarray: Adjusted pose with enforced bone lengths.
    """
    pose_corrected = pose.copy()
    # Loop through bones
    for parent, child in skeleton:
        # Get joint positions
        p_pos = pose_corrected[parent]
        c_pos = pose_corrected[child]
        # current bone length
        vec = c_pos - p_pos
        current_len = np.linalg.norm(vec)
        # skip invalid bones
        if current_len < 1e-6:
            continue
        # get target bone length from canonical lengths, or use current if missing
        target_len = canonical_bone_lengths.get((parent, child), current_len)
        scale = target_len / current_len
        # blend between current and target length
        corrected_vec = vec * (1 + (scale - 1) * blend)
        # move child to correct distance from parent
        pose_corrected[child] = p_pos + corrected_vec
    return pose_corrected


def is_valid_pair(conf, p, c, conf_thresh=0.3):
    """
    Check if a bone (p → c) is reliable enough to use
    Only use bones that are trustworthy for depth estimation

    Args:
        kp2d: 2D keypoints (x, y)
        conf: confidence score for each joint
        p, c: parent and child joint indices
        conf_thresh = 0.3 → minimum confidence

    Returns:
        bool: True if both joints are confident enough, False otherwise
    """
    # If no confidence model exists, assume all joints are valid
    if conf is None:
        return True
    # Both joints must be confident
    return conf[p] > conf_thresh and conf[c] > conf_thresh


def estimate_z_for_frame(kp2d_frame, conf_frame, K, canonical_bone_lengths, bone_groups):
    """
    Z depth of the person in camera space

    Args:
        keypoints_2d: (J,2) array of 2D joint positions
        confidence: (J,) array of confidence scores for each joint
        K: (3,3) camera intrinsic matrix
        canonical_bone_lengths: dict of target bone lengths
        bone_groups: list of lists of (parent, child) pairs to try for depth estimation

    Returns:
        float: Estimated Z depth of the root joint
    """
    # average focal length
    f = (K[0,0] + K[1,1]) / 2
    # Try different body regions (legs, arms, torso)
    for group in bone_groups:
        # Store multiple depth estimates
        z_vals = []
        # Check each bone (parent and child) in the group
        for (p, c) in group:
            # Skip if confidence is too low for either joint
            if conf_frame is not None:
                if conf_frame[p] < 0.4 or conf_frame[c] < 0.4:
                    continue
            # 2D distance between parent and child joints
            d2d = np.linalg.norm(kp2d_frame[c] - kp2d_frame[p])
            # Skip if 2D distance is too small (unreliable)
            if d2d < 5.0:
                continue
            
            # Corresponding 3D bone length from canonical model
            d3d = canonical_bone_lengths.get((p, c))
            if d3d is None:
                continue
            # Estimate Z using: depth ≈ (f * real_size) / image_size
            return f * d3d / d2d  # first valid bone wins

    return None  # failed

def backproject_root(kp_root_2d, Z, K, R_cam, t_cam):
    """
    Compute 3D root position directly from 2D + depth

    Args:
        kp_root_2d: (2,) pixel coordinates of root joint
        Z: scalar depth value for root joint in camera space
        K: (3,3) camera intrinsic matrix
        R_cam: (3,3) camera rotation matrix
        t_cam: (3,) camera translation vector

    Returns:
        root_world (3,)
    """
    # Camera intrinsics 
    # fx, fy = focal length
    fx = K[0, 0]
    fy = K[1, 1]
    # cx, cy = image center (principal point)
    cx = K[0, 2]
    cy = K[1, 2]
    # pixel location of the root joint
    u, v = kp_root_2d
    # Back-project to camera space: given (u,v) and Z, find (X,Y,Z) in camera space
    Xc = (u - cx) / fx * Z
    Yc = (v - cy) / fy * Z
    Zc = Z
    # 3D point in camera coordinates
    root_cam = np.array([Xc, Yc, Zc])
    # Convert to world space
    root_world = R_cam.T @ (root_cam - t_cam)

    return root_world


def compute_root_rotation_maya(pose_local, root_idx=0):
    """
    Compute root rotation from first frame in Maya coordinates.

    Args:
        pose_local: (J,3) np.ndarray, root-relative 3D joint positions for first frame
        root_idx: int, index of the hip/root joint (usually 0)

    Returns:
        R_root: (3,3) np.ndarray, rotation matrix for root (yaw + pitch)
        yaw_deg: float, yaw angle in degrees
        pitch_deg: float, pitch angle in degrees
    """
    # root joint (center of body)
    hip = pose_local[root_idx]

    # Use shoulder vector for yaw
    lshoulder = pose_local[11] - hip  # left shoulder
    rshoulder = pose_local[14] - hip  # right shoulder

    # flatten to ground plane (ignore vertical)
    forward_vec = lshoulder - rshoulder  # left minus right
    forward_vec[1] = 0
    # If shoulders are basically at the same position, assume person is facing forward (Z direction)
    if np.linalg.norm(forward_vec) < 1e-6:
        forward_vec = np.array([0,0,1])
    else:
        # normalize vector
        forward_vec /= np.linalg.norm(forward_vec)
    # convert shoulder direction into a horizontal rotation angle
    yaw = np.arctan2(forward_vec[0], forward_vec[2])

    # Use spine vector for pitch (lean forward/back)
    spine = pose_local[8] - hip  # thorax relative to hip
    # distance in XZ plane
    horizontal_dist = np.linalg.norm(spine[[0, 2]])
    # negative = lean forward, positive = lean backward
    pitch = np.arctan2(-spine[1], horizontal_dist)  

    # Build rotation matrix
    # Ry = rotate left/right
    Ry = R.from_euler('y', yaw).as_matrix()  # yaw around Y
    # Rx = tilt forward/back
    Rx = R.from_euler('x', pitch).as_matrix()  # pitch around X
    R_root = Ry @ Rx

    return R_root, np.degrees(yaw), np.degrees(pitch)


# Constrains
def compute_angle(a, b, c):
    """
    Calc the interior angle between three joints
    """
    # Vector from joint to parent and child
    v1 = a - b
    v2 = c - b
    # Normalize vectors (make them length 1) to ignore bone length
    v1 /= (np.linalg.norm(v1) + 1e-8)
    v2 /= (np.linalg.norm(v2) + 1e-8)
    # Dot product + arccos gives the angle in radians
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    # Convert to degrees
    return np.degrees(np.arccos(dot))

def joint_angle_penalty(pose):
    """
    Ensure joints don't bend past human limits
    """
    penalty = 0.0
    for (p, j, c), (min_a, max_a) in JOINT_ANGLE_LIMITS.items():
        angle = compute_angle(pose[p], pose[j], pose[c])
        # If the angle is outside [min_a, max_a], adds the squared difference to the penalty
        if angle < min_a:
            delta = min_a - angle
        elif angle > max_a:
            delta = angle - max_a
        else:
            continue
        # clamp linear error first
        delta = min(delta, 20.0)
        # then square
        penalty += delta * delta
    return penalty

def hinge_direction_penalty(pose):
    """
    Use a dot product to check if the lower leg is bending forward relative to the thigh
    """
    penalty = 0.0
    # Right knee
    if np.dot(pose[2]-pose[1], pose[3]-pose[2]) < 0:
        penalty += 100.0
    # Left knee
    if np.dot(pose[5]-pose[4], pose[6]-pose[5]) < 0:
        penalty += 100.0
    return penalty

def shoulder_constraint_penalty(pose):
    """
    Penalizes if the left shoulder is higher/lower than the right
    Forces the thorax to stay centered between the shoulders
    Ensure the line from left-shoulder to right-shoulder is straight
    If they aren't collinear, the cross product magnitude increases
    """
    thorax = pose[8]
    l_sh   = pose[11]
    r_sh   = pose[14]
    penalty = 0.0
    # Force shoulders to be level (same Y value)
    penalty += 10.0 * (l_sh[1] - r_sh[1])**2
    # Force thorax to be centered between shoulders
    mid = 0.5 * (l_sh + r_sh)
    penalty += 20.0 * np.linalg.norm(thorax - mid)**2
    # Force shoulders to be in a straight line (collinear)
    v1 = l_sh - thorax
    v2 = r_sh - thorax

    v1_n = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_n = v2 / (np.linalg.norm(v2) + 1e-8)

    penalty += 20.0 * np.linalg.norm(np.cross(v1_n, v2_n))**2
    # Symmetry
    penalty += 5.0 * (np.linalg.norm(v1) - np.linalg.norm(v2))**2
    return penalty

def hip_constraint_penalty(pose):
    """
    Penalizes if the left hip is higher/lower than the right
    Forces the root to stay centered between the hips
    Ensure the line from left-hip to right-hip is straight
    If they aren't collinear, the cross product magnitude increases
    """
    root = pose[0]
    r_hip = pose[1]
    l_hip = pose[4]

    vR = r_hip - root
    vL = l_hip - root

    vR_n = vR / (np.linalg.norm(vR) + 1e-8)
    vL_n = vL / (np.linalg.norm(vL) + 1e-8)

    penalty = 0.0
    penalty += np.linalg.norm(vR_n + vL_n)**2
    penalty += np.linalg.norm(np.cross(vR_n, vL_n))**2
    penalty += 0.1 * (np.linalg.norm(vR) - np.linalg.norm(vL))**2

    return penalty

# Neck+Head Constrains
def neck_head_constraint_penalty(pose):
    """
    Enforce a natural neck/head position by ensuring the head is above the neck, the neck is above the thorax, and all three are roughly aligned.
    """
    thorax = pose[8]
    neck   = pose[9]
    head   = pose[10]
    penalty = 0.0
    # Vertical ordering (Y axis)
    # head should be above neck
    if head[1] < neck[1]:
        penalty += (neck[1] - head[1])**2
    # neck should be above thorax
    if neck[1] < thorax[1]:
        penalty += (thorax[1] - neck[1])**2
    # Collinearity (straight spine)
    v1 = neck - thorax
    v2 = head - neck
    v1_n = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_n = v2 / (np.linalg.norm(v2) + 1e-8)
    # penalize bending
    penalty += np.linalg.norm(np.cross(v1_n, v2_n))**2
    return penalty

def refine_pose_with_2d_ik_partial(
    pose_world,
    kp_2d,
    K, R_cam, t_cam,
    skeleton,
    canonical_bone_lengths,
    optimize_joints,
    confidence=None
):
    """
    Refine the pose by optimizing only a subset of joints to better match 2D keypoints while enforcing bone lengths.

    Args:
        pose_world: (J,3) initial 3D joint positions in world space
        kp_2d: (J,2) 2D keypoints to match
        K, R_cam, t_cam: camera parameters for projection
        skeleton: list of (parent, child) pairs defining the skeleton
        canonical_bone_lengths: dict mapping (parent, child) to target bone length
        optimize_joints: list of joint indices to optimize
        confidence: (J,) array of confidence scores for each joint (optional)

    Returns:
        pose_world_refined: (J,3) refined 3D joint positions in world space
    """
    pose = pose_world.copy()

    # Flatten only optimizable joints
    x0 = pose[optimize_joints].flatten()

    def reconstruct_pose(x):
        new_pose = pose.copy()
        # only replace the joints being optimized
        new_pose[optimize_joints] = x.reshape(-1, 3)
        return new_pose

    def loss(x):
        pose_full = reconstruct_pose(x)
        # Project 3D to 2D
        proj, valid = project_points(pose_full, K, R_cam, t_cam)
        reproj_err = 0.0
        for j in optimize_joints:  # only care about these joints
            if not valid[j]:
                continue
            w = 1.0
            if confidence is not None:
                w = confidence[j]
            # projected 3D vs actual 2D keypoint
            reproj_err += w * np.linalg.norm(proj[j] - kp_2d[j])**2

        # Bone length constrain
        bone_err = 0.0
        for (p, c) in skeleton:
            d = np.linalg.norm(pose_full[c] - pose_full[p])
            target = canonical_bone_lengths[(p, c)]
            bone_err += (d - target)**2
        
        # legal pose constrains
        angle_err    = joint_angle_penalty(pose_full)
        hinge_err    = hinge_direction_penalty(pose_full)
        shoulder_err = shoulder_constraint_penalty(pose_full)
        hip_err      = hip_constraint_penalty(pose_full)
        neck_err = neck_head_constraint_penalty(pose_full)

        # combine losses with a weight to enforce bone lengths more strongly
        total = (
            3.0 * reproj_err
            + 10.0 * bone_err
            + 0.2 * angle_err
            + hinge_err
            + 3.0 * shoulder_err
            + 3.0 * hip_err
            + 3.0 * neck_err
        )

        # Print debug (per evaluation)
        # print(
        #     f"[loss] reproj={reproj_err:.3f} | "
        #     f"bone={bone_err:.3f} | "
        #     f"angle={angle_err:.3f} | "
        #     f"hinge={hinge_err:.3f} | "
        #     f"shoulder={shoulder_err:.3f} | "
        #     f"hip={hip_err:.3f} | "
        #     f"neck={neck_err:.3f} | "
        #     f"TOTAL={total:.3f}"
        # )

        return total
    # Find values of x that make loss(x) as small as possible
    res = minimize(
        loss,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 15}
    )
    # Return improved pose
    return reconstruct_pose(res.x)

def check_bone_length_consistency(poses, skeleton):
    """
    Check bone length consistency across frames and print statistics.
    Args:
        poses: (F, J, 3) array of 3D joint positions across frames    
        skeleton: list of (parent, child)
    """
    print("\nBone Length Consistency Check:")
    # Loop through each bone (parent and child) in the skeleton
    for (p, c) in skeleton:
        lengths = np.linalg.norm(poses[:, p] - poses[:, c], axis=1)
        mean_len = np.mean(lengths)
        std_len  = np.std(lengths)
        min_len  = np.min(lengths)
        max_len  = np.max(lengths)

        print(f"Bone {p}->{c} | mean={mean_len:.2f} cm | std={std_len:.2f} | range=[{min_len:.2f}, {max_len:.2f}]")
        # Flag unstable bones
        if std_len > 2.0:  # threshold (tune if needed)
            print("WARNING: Length not stable!")


def compute_z_values(F, keypoints_2d, confidence, K, canonical_bone_lengths, bone_groups):
    """
    Estimate per-frame Z depth at keyframes and interpolate the rest.
 
    Args:
        F (int): Total number of frames.
        keypoints_2d (np.ndarray): Shape (F, J, 2).
        confidence (np.ndarray or None): Shape (F, J).
        K (np.ndarray): Camera intrinsics.
        canonical_bone_lengths (dict): Target bone lengths.
        bone_groups (list): Groups of bones used for depth estimation.

 
    Returns:
        np.ndarray: Shape (F,), interpolated Z depth values.
    """
    Z_raw = np.zeros(F)

    for i in range(F):
        conf_i = confidence[i] if confidence is not None else None
        z = estimate_z_for_frame(keypoints_2d[i], conf_i, K, canonical_bone_lengths, BONE_GROUPS)
        if z is not None:
            Z_raw[i] = z

    # # Fill zeros with neighbour average
    # valid_z = Z_raw[Z_raw > 0]
    # fallback = np.median(valid_z) if len(valid_z) > 0 else 200.0
    # for i in range(F):
    #     if Z_raw[i] == 0:
    #         Z_raw[i] = fallback

    # Smooth — depth changes slowly
    Z_values = uniform_filter1d(Z_raw, size=15)

    print(f"Z depth range(Depth between camera and character): {Z_values.min():.1f} – {Z_values.max():.1f} cm")
    print(f"Z depth std/mean: {Z_values.std()/Z_values.mean():.3f}  (good if < 0.15)")
    return Z_values

def compute_root_world_positions(F, keypoints_2d, Z_values, K, R_frames, t_frames):
    """
    Back-project 2D root keypoints to 3D world positions using estimated Z.
 
    Args:
        F (int): Number of frames.
        keypoints_2d (np.ndarray): Shape (F, J, 2).
        Z_values (np.ndarray): Shape (F,), per-frame root depth.
        K (np.ndarray): Camera intrinsics.
        R_frames (np.ndarray): Shape (F, 3, 3), per-frame camera rotations.
        t_frames (np.ndarray): Shape (F, 3), per-frame camera translations.
 
    Returns:
        np.ndarray: Shape (F, 3), root positions in world coordinates.
    """
    root_3d_world = []
    print ("Z depth for every 5 frames:")
    for i in range(F):
        
        kp_root_2d = keypoints_2d[i, 0, :2]

        z = Z_values[i]

        root_world = backproject_root(
            kp_root_2d,
            z,
            K,
            R_frames[i],
            t_frames[i]
        )

        root_3d_world.append(root_world)
        if i % 5 == 0:
            print(f"Frame {i:03d} | Z = {z:.2f}")
    
    return np.array(root_3d_world)

def solve_world_poses(frames, poses_3d, keypoints_2d, confidence, K, R_frames, t_frames, F, canonical_bone_lengths):
    """
    Full solve: estimate depth → assemble world pose → enforce bones → IK refine.
 
    Args:
        frames (list): RGB frames (unused in solve, passed through for convenience).
        poses_3d (np.ndarray): Shape (F, J, 3), camera-space poses.
        keypoints_2d (np.ndarray): Shape (F, J, 2).
        confidence (np.ndarray or None): Shape (F, J).
        K (np.ndarray): Camera intrinsics.
        R_frames (np.ndarray): Shape (F, 3, 3).
        t_frames (np.ndarray): Shape (F, 3).
        F (int): Number of frames.
        canonical_bone_lengths (dict): Target bone lengths.
 
    Returns:
        np.ndarray: Shape (F, J, 3), refined world-space poses.
    """
    # Root-relative pose
    root_pos = poses_3d[:, 0:1, :]
    poses_3d_local = poses_3d - root_pos

    # Root rotation from first frame
    R_root_first, yaw_deg, pitch_deg = compute_root_rotation_maya(poses_3d_local[0])
    print(f"Frame 0 root rotation: yaw={yaw_deg:.2f}°, pitch={pitch_deg:.2f}°")

    # Repeat for all frames
    F = poses_3d_local.shape[0]
    root_rotations = np.repeat(R_root_first[None, :, :], F, axis=0)

    # Depth estimation
    Z_values = compute_z_values(
        F, keypoints_2d, confidence, K, canonical_bone_lengths, BONE_GROUPS
    )

    # Back-project root to world space
    root_3d_world = compute_root_world_positions(
        F, keypoints_2d, Z_values, K, R_frames, t_frames
    )

    # Assemble world pose
    poses_3d_world = np.zeros_like(poses_3d)
    for i in range(F):
        # Rotate local (camera-space) offsets into world space
        local_world = (R_frames[i].T @ poses_3d_local[i].T).T
        poses_3d_world[i] = local_world + root_3d_world[i]

    # Per-frame bone enforcement + IK refinement
    for i in range(F):
        poses_3d_world[i] = enforce_bone_lengths(
            poses_3d_world[i],
            SKELETON,
            canonical_bone_lengths,
            blend=1.0   # full correction
        )

        poses_3d_world[i] = refine_pose_with_2d_ik_partial(
        poses_3d_world[i],
        keypoints_2d[i],
        K,
        R_frames[i],
        t_frames[i],
        SKELETON,
        canonical_bone_lengths,
        OPTIMIZE_JOINTS,
        confidence[i] if confidence is not None else None
        )
    return poses_3d_world
