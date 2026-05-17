import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from global_motion_recovery import SKELETON, project_points

# VideoPose3D preview
 
def make_videopose3d_update_fn(frames, predicted_3d, ax_video, ax_3d):
    """
    Build and return the per-frame update function for the VideoPose3D preview.
 
    Args:
        frames (list of np.ndarray): RGB frames.
        predicted_3d (np.ndarray): Shape (F, J, 3), raw VideoPose3D output.
        ax_video (Axes): Left panel for the RGB frame.
        ax_3d (Axes3D): Right panel for the 3D skeleton.
 
    Returns:
        Callable: update(frame) → None.
    """
    def update(frame):
        ax_video.clear()
        ax_3d.clear()
 
        ax_video.imshow(frames[frame])
        ax_video.axis("off")
 
        pose = predicted_3d[frame]
        x =  pose[:, 0]
        y = -pose[:, 2]
        z = -pose[:, 1]
 
        ax_3d.scatter(x, y, z)
        for j1, j2 in SKELETON:
            ax_3d.plot([x[j1], x[j2]], [y[j1], y[j2]], [z[j1], z[j2]])
 
        ax_3d.set_xlim(-1, 1)
        ax_3d.set_ylim(-1, 2)
        ax_3d.set_zlim(-1, 1)
        ax_3d.view_init(elev=15, azim=frame * 2)
 
    return update
 
 
def run_videopose3d_preview(frames, predicted_3d, output_path):
    """
    Render and save the VideoPose3D side-by-side preview (video + 3D skeleton).
 
    Args:
        frames (list of np.ndarray): RGB frames.
        predicted_3d (np.ndarray): Shape (F, J, 3), raw VideoPose3D output.
        output_path (str): Full path for the saved video (e.g. .../pose_3d.mp4).
    """
    num_frames = min(len(frames), predicted_3d.shape[0])
 
    fig      = plt.figure(figsize=(10, 5))
    ax_video = fig.add_subplot(1, 2, 1)
    ax_3d    = fig.add_subplot(1, 2, 2, projection="3d")
 
    update = make_videopose3d_update_fn(frames, predicted_3d, ax_video, ax_3d)
 
    ani = FuncAnimation(fig, update, frames=num_frames, interval=40)
    ani.save(output_path, writer="ffmpeg")
    plt.close(fig)
    print(f"Saved 3D preview to {output_path}")
 
# World-space skeleton animation
def build_figure(frames, poses_3d_world):
    """
    Create the matplotlib figure and initial artists for the animation.

    Args:
        frames (list of np.ndarray): RGB frames.
        poses_3d_world (np.ndarray): Shape (F, J, 3), world-space poses.

    Returns:
        fig (Figure)
        ax2d (Axes): 2D image panel.
        ax3d (Axes3D): 3D skeleton panel.
        im2d (AxesImage): Image artist.
        joints2d_scatter (PathCollection): All 2D joint dots.
        root_dot (PathCollection): Root joint highlight.
        lines3d (list): 3D bone line artists.
        root_dot_3d (Path3DCollection): 3D root joint highlight.
    """
    fig  = plt.figure(figsize=(16, 8))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    # Make the 2D panel take more horizontal space
    ax2d.set_position([0.02, 0.05, 0.55, 0.90])   # [left, bottom, width, height]
    ax3d.set_position([0.60, 0.05, 0.38, 0.90])
    # 2D panel
    im2d             = ax2d.imshow(frames[0])
    h, w = frames[0].shape[:2]
    ax2d.set_xlim(0, w)
    ax2d.set_ylim(h, 0)   # flipped because image Y goes top-down
    ax2d.set_aspect('equal')
    ax2d.set_title('2D Keypoints Overlay')
    joints2d_scatter = ax2d.scatter([], [], c='red',    s=20)
    root_dot         = ax2d.scatter([], [], c='yellow', s=50)

    # 3D panel
    lines3d     = [ax3d.plot([], [], [], 'o-', lw=2, color='blue')[0] for _ in SKELETON]
    root_dot_3d = ax3d.scatter([], [], [], c='yellow', s=50)

    ax3d.set_xlim(np.min(poses_3d_world[:, :, 0]) - 50, np.max(poses_3d_world[:, :, 0]) + 50)
    ax3d.set_ylim(-np.max(poses_3d_world[:, :, 2]) - 50, -np.min(poses_3d_world[:, :, 2]) + 50)
    ax3d.set_zlim(np.min(poses_3d_world[:, :, 1]) - 50, np.max(poses_3d_world[:, :, 1]) + 50)
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Z')
    ax3d.set_zlabel('Y (up)')
    ax3d.set_title('3D Skeleton')

    return fig, ax2d, ax3d, im2d, joints2d_scatter, root_dot, lines3d, root_dot_3d


def make_update_fn(
    frames, poses_3d_world, keypoints_2d,
    K, R_frames, t_frames,
    ax2d, im2d, joints2d_scatter, root_dot,
    lines3d, root_dot_3d,
):
    """
    Build and return the per-frame animation update function.

    Args:
        frames (list): RGB frames.
        poses_3d_world (np.ndarray): Shape (F, J, 3).
        keypoints_2d (np.ndarray): Shape (F, J, 2).
        K (np.ndarray): Camera intrinsics.
        R_frames (np.ndarray): Shape (F, 3, 3).
        t_frames (np.ndarray): Shape (F, 3).
        ax2d (Axes): 2D panel (needed to create skeleton lines on first frame).
        im2d, joints2d_scatter, root_dot: 2D artists.
        lines3d, root_dot_3d: 3D artists.

    Returns:
        Callable: update(frame_idx) → list of updated artists.
    """
    skeleton_lines_2d = []

    def update(frame_idx):
        # Update 2D image
        im2d.set_data(frames[frame_idx])
        h, w = frames[frame_idx].shape[:2]
        ax2d.set_xlim(0, w)
        ax2d.set_ylim(h, 0)   # keep locked every frame
        # Update 2D keypoint dots
        root_dot.set_offsets([keypoints_2d[frame_idx, 0, :2]])
        joints2d_scatter.set_offsets(keypoints_2d[frame_idx, :, :2])

        # Project 3D skeleton into 2D and draw
        pose_world = poses_3d_world[frame_idx]
        proj, _    = project_points(pose_world, K, R_frames[frame_idx], t_frames[frame_idx])

        if len(skeleton_lines_2d) == 0:
            for (p, c) in SKELETON:
                line, = ax2d.plot(
                    [proj[p, 0], proj[c, 0]],
                    [proj[p, 1], proj[c, 1]],
                    'o-', lw=2, color='cyan'
                )
                skeleton_lines_2d.append(line)
        else:
            for line, (p, c) in zip(skeleton_lines_2d, SKELETON):
                line.set_data([proj[p, 0], proj[c, 0]], [proj[p, 1], proj[c, 1]])

        # Update 3D skeleton
        for line, (p, c) in zip(lines3d, SKELETON):
            line.set_data([pose_world[p, 0], pose_world[c, 0]],
                        [-pose_world[p, 2], -pose_world[c, 2]])  # negated
            line.set_3d_properties([pose_world[p, 1], pose_world[c, 1]])

        root_dot_3d._offsets3d = (
            [pose_world[0, 0]], [-pose_world[0, 2]], [pose_world[0, 1]]  # negated
        )

        return [im2d, root_dot, joints2d_scatter, root_dot_3d] + lines3d + skeleton_lines_2d

    return update


def run_animation(frames, poses_3d_world, keypoints_2d, K, R_frames, t_frames, interval=100, output_path=None):
    """
    Build the figure, wire up the animation, optionally save it, then show it.

    Args:
        frames (list): RGB frames.
        poses_3d_world (np.ndarray): Shape (F, J, 3).
        keypoints_2d (np.ndarray): Shape (F, J, 2).
        K (np.ndarray): Camera intrinsics.
        R_frames (np.ndarray): Shape (F, 3, 3).
        t_frames (np.ndarray): Shape (F, 3).
        interval (int): Milliseconds between frames.
        output_path (str or None): If provided, saves the animation to this path.
            Use a .mp4 extension (requires ffmpeg) or .gif (requires pillow).
    """
    F = poses_3d_world.shape[0]

    fig, ax2d, ax3d, im2d, joints2d_scatter, root_dot, lines3d, root_dot_3d = build_figure(
        frames, poses_3d_world
    )

    update = make_update_fn(
        frames, poses_3d_world, keypoints_2d,
        K, R_frames, t_frames,
        ax2d, im2d, joints2d_scatter, root_dot,
        lines3d, root_dot_3d,
    )

    ani = FuncAnimation(fig, update, frames=F, interval=interval)

    if output_path is not None:
        ext = os.path.splitext(output_path)[1].lower()
        writer = "pillow" if ext == ".gif" else "ffmpeg"
        ani.save(output_path, writer=writer)
        print(f"Saved animation to {output_path}")

    plt.show()