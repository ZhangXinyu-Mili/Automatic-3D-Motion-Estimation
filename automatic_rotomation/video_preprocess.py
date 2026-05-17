import os
import cv2
import sys
import re
import shutil

def preprocess(project_folder):
    # Ask user for input folder
    input_folder = input("Enter the path to the image sequence folder: ").strip()
    # Remove quotation marks if user pasted a path with quotes
    input_folder = input_folder.strip('"').strip("'")
    # Check if folder exists
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Error: Folder '{input_folder}' does not exist.")
    print(f"Input folder set to: {input_folder}")
    # Validate input folder
    files = sorted(os.listdir(input_folder))
    if len(files) == 0:
        raise ValueError("Error: Input folder is empty.")

    # Output folder
    output_folder = os.path.join(project_folder, "preprocessed")
    # Check if output folder exists
    if os.path.exists(output_folder):
        user_input = input(
            f"Output folder '{output_folder}' exists. Would you like to overwrite? (y/n): "
        ).lower()

        if user_input in ["y", "yes"]:
            shutil.rmtree(output_folder)
            os.makedirs(output_folder)
            print(f"Cleared existing folder: {output_folder}")
        else:
            raise RuntimeError("Progress has been cancelled.")
    else:
        os.makedirs(output_folder)
        print(f"Created preprocessed image sequeence folder: {output_folder}")

    # Resize settings
    TARGET_WIDTH = 640
    TARGET_HEIGHT = 384
    # Allowed file extensions
    ALLOWED_EXT = [".jpg", ".jpeg"]
    VIDEO_EXT = [".mp4", ".mov", ".avi", ".mkv"]

    # Detect image sequence
    frame_dict = {}
    prefix = None
    extension = None
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in VIDEO_EXT:
            raise ValueError(f"Error: Video file detected ({f}). Please convert it into .jpg format. Only .jpg allowed.")
        if ext not in ALLOWED_EXT:
            print(f"Skipping unsupported file: {f}")
            continue
        match = re.search(r"(.*?)(\d+)(\.[^.]+)$", f)
        if not match:
            continue
        prefix = match.group(1)
        frame_number = int(match.group(2))
        extension = match.group(3)
        frame_dict[frame_number] = f
    if len(frame_dict) == 0:
        raise ValueError("Error: No valid image sequence detected.")
    
    # Print detected sequence
    frames = sorted(frame_dict.keys())
    # Check for missing frames
    missing = []
    for i in range(len(frames) - 1):
        if frames[i+1] != frames[i] + 1:
            missing.extend(range(frames[i] + 1, frames[i+1]))
    if missing:
        raise ValueError(f"Error: Missing frames detected: {missing[:10]}{'...' if len(missing)>10 else ''}")
    
    example_file = frame_dict[frames[0]]
    match_example = re.search(r'(\d+)', example_file)
    padding = len(match_example.group(1))
    print("\nSequence detected:")
    print(f"{prefix}%0{padding}d{extension}")
    print(f"Frame range: {frames[0]} → {frames[-1]}")

    # Ask user for frame range
    while True:
        start_input = input("Enter start frame: ").strip()
        end_input = input("Enter end frame: ").strip()
        # Check if numbers
        if not start_input.isdigit() or not end_input.isdigit():
            print("Error: Frame numbers must be integers. Please try again.\n")
            continue
        START_FRAME = int(start_input)
        END_FRAME = int(end_input)
        # Check order
        if START_FRAME > END_FRAME:
            print("Error: Start frame cannot be greater than end frame.\n")
            continue
        # Check range
        if START_FRAME < frames[0] or END_FRAME > frames[-1]:
            print("Error: Frame range outside the available sequence.\n")
            continue
        break

    print(f"\nProcessing frames {START_FRAME} → {END_FRAME}")

    # Process frames
    processed = 0
    for frame in frames:
        if frame < START_FRAME or frame > END_FRAME:
            continue
        filename = frame_dict[frame]
        path = os.path.join(input_folder, filename)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # Check if image was read successfully
        if img is None or img.size == 0:
            raise ValueError(f"Error: Corrupted image detected: {filename}")
        # Resize image to target resolution
        resized = cv2.resize(
            img,
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )
        # Save preprocessed image
        output_name = f"{prefix}{frame:0{padding}d}.jpg"
        output_path = os.path.join(output_folder, output_name)
        cv2.imwrite(output_path, resized)

        processed += 1

    print("\nPreprocessing complete.")
    print("Frames processed:", processed)
    print("Output directory:", output_folder)

    # return for pipeline
    return output_folder, START_FRAME, END_FRAME

