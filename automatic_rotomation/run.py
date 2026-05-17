import os
import numpy as np
import shutil
from video_preprocess import preprocess
from detection_process import run_videopose3d, run_yolo_pose_tracking
from global_motion_recovery import load_frames, get_canonical_bone_lengths, solve_world_poses, check_bone_length_consistency, project_points, SKELETON
from display_plots import run_animation
def main():
    # Set project folder
    project_folder = input("Set the path to the project folder: ").strip()
    project_folder = project_folder.strip('"').strip("'")
    # If project_folder doesn't exist
    if not os.path.exists(project_folder):
        user_input = input(
            f"Folder '{project_folder}' does not exist. Would you like to create it? (y/n): "
        ).strip().lower()
        # Create the folder when user said yes
        if user_input in ["y", "yes"]:
            os.makedirs(project_folder, exist_ok=True)
            print(f"Created project folder: {project_folder}")
        else:
            raise RuntimeError("Cancelled.")
    else:
        print(f"Project folder set to: {project_folder}")
    # Step 1: Preprocess video frames
    image_folder, START_FRAME, END_FRAME  = preprocess(project_folder)
    # Step 2: Run YOLO pose tracking
    keypoints_norm, frames, keypoints_2d, confidence = run_yolo_pose_tracking(image_folder, project_folder)
    # Ask user for actor's approx height
    while True:
        user_input = input("Enter the character's height (cm): ").strip()
        try:
            USER_HEIGHT_CM = float(user_input)
        except ValueError:
            print("Error: Please enter a valid number (e.g., 170 or 170.5).")
            continue
        if USER_HEIGHT_CM < 50 or USER_HEIGHT_CM > 300:
            print("Error: Height must be between 50 and 300 cm.")
            continue
        break
    # Step 3: Run VideoPose3D to predict 3D pose and render video
    predicted_3d = run_videopose3d(
        keypoints_2d=keypoints_norm,
        frames=frames,
        project_folder=project_folder,
        videopose3d_path=os.path.join(os.path.dirname(__file__), "VideoPose3D-main"),
        checkpoint_path=os.path.join(os.path.dirname(__file__), "VideoPose3D-main", "checkpoint", "pretrained_h36m_cpn.bin")
        )

    # Step 4: Camera file input + copy into project folder
    camera_path_in_project = os.path.join(project_folder, "camera_data.json")

    # Case 1: camera already exists in project folder
    if os.path.exists(camera_path_in_project):
        print(f"\nFound existing camera file: {camera_path_in_project}")

        while True:
            choice = input(
                "Use existing camera file? (y) or provide new one? (n): "
            ).strip().lower()

            if choice in ["y", "yes"]:
                camera_json = camera_path_in_project
                print("Using existing camera file.")
                break

            elif choice in ["n", "no"]:
                # Ask for new file
                while True:
                    new_camera = input("Enter path to new camera .json file: ").strip()
                    new_camera = new_camera.strip('"').strip("'")

                    if not os.path.exists(new_camera):
                        print("Error: File does not exist. Try again.\n")
                        continue

                    if not new_camera.endswith(".json"):
                        print("Error: Please provide a .json file.\n")
                        continue

                    shutil.copy2(new_camera, camera_path_in_project)
                    print(f"Replaced camera file in project folder: {camera_path_in_project}")
                    camera_json = camera_path_in_project
                    break

                break

            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    # Case 2: no camera file exists
    else:
        print("\nNo camera file found in project folder.")

        while True:
            new_camera = input("Enter path to camera .json file: ").strip()
            new_camera = new_camera.strip('"').strip("'")

            if not os.path.exists(new_camera):
                print("Error: File does not exist. Try again.\n")
                continue

            if not new_camera.endswith(".json"):
                print("Error: Please provide a .json file.\n")
                continue

            shutil.copy2(new_camera, camera_path_in_project)
            print(f"Saved camera file to project folder: {camera_path_in_project}")
            camera_json = camera_path_in_project
            break

    # Step 5: Solve world-space poses
    output_npy = os.path.join(project_folder, "predicted_3d_world.npy")

    frames, _, _, poses_3d, K, R_frames, t_frames, F = load_frames(
        image_folder, predicted_3d, camera_json, START_FRAME, END_FRAME)
 
    canonical_bone_lengths = get_canonical_bone_lengths(USER_HEIGHT_CM)
 
    poses_3d_world = solve_world_poses(
        frames, poses_3d, keypoints_2d, confidence,
        K, R_frames, t_frames, F, canonical_bone_lengths
    )
    # Diagnostics
    check_bone_length_consistency(poses_3d_world, SKELETON)
    np.save(output_npy, poses_3d_world)
    print(f"Saved world skeleton to {output_npy}")
    
    # Step 5: Visualise 
    run_animation(frames, poses_3d_world, keypoints_2d, K, R_frames, t_frames, output_path=os.path.join(project_folder, "predicted_3d_world.mp4"))
    print("\nPipeline complete!")

if __name__ == "__main__":
    main()