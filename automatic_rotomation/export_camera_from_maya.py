import maya.cmds as cmds
import json
import numpy as np


# def export_camera_data(camera_name, start_frame, end_frame, output_path):
#     cam_shape = cmds.listRelatives(camera_name, shapes=True)[0]

#     # ── Intrinsics (unchanged) ────────────────────────────────────────────
#     focal_length  = cmds.getAttr(f"{cam_shape}.focalLength")
#     h_aperture    = cmds.getAttr(f"{cam_shape}.horizontalFilmAperture") * 25.4
#     v_aperture    = cmds.getAttr(f"{cam_shape}.verticalFilmAperture")   * 25.4
#     render_width  = cmds.getAttr("defaultResolution.width")
#     render_height = cmds.getAttr("defaultResolution.height")
#     fx = (focal_length / h_aperture) * render_width
#     fy = (focal_length / v_aperture) * render_height
#     cx = render_width  / 2.0
#     cy = render_height / 2.0
#     K  = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

#     flip = np.diag([1.0, -1.0, -1.0])

#     # ── Pass 1: collect all cam-to-world matrices ─────────────────────────
#     R_c2w_list   = []
#     t_c2w_list   = []   # camera world positions (cm)

#     for f in range(start_frame, end_frame + 1):
#         cmds.currentTime(f)
#         m = cmds.getAttr(f"{camera_name}.worldMatrix[0]")
#         M = np.array(m).reshape(4, 4).T   # column-major → row-major

#         # FIX: worldMatrix bakes in camera scale → orthonormalise via SVD
#         # On cameras without scale this is a no-op; on scaled cameras it
#         # strips the scale and leaves a pure rotation matrix.
#         U, _, Vt = np.linalg.svd(M[:3, :3])
#         R_c2w    = U @ Vt
#         if np.linalg.det(R_c2w) < 0:      # ensure proper rotation (det=+1)
#             U[:, -1] *= -1
#             R_c2w = U @ Vt

#         R_c2w_list.append(R_c2w)
#         t_c2w_list.append(M[:3, 3])       # camera position in Maya world

#     # ── Re-centre on frame-0 camera position ─────────────────────────────
#     # Prevents t values from being millions of cm when the scene origin is
#     # far from the camera. videopose3d_to_hik.py adds this back when loading.
#     world_origin = t_c2w_list[0].copy()

#     # ── Pass 2: compute world-to-camera extrinsics ────────────────────────
#     frames = []
#     for i, f in enumerate(range(start_frame, end_frame + 1)):
#         R_c2w  = R_c2w_list[i]
#         cam_pos = t_c2w_list[i]   # ✅ NO recentering
    
#         R_w2c  = R_c2w.T
#         t_w2c  = -R_w2c @ cam_pos
    
#         R_cv   = flip @ R_w2c
#         t_cv   = flip @ t_w2c
    
#         frames.append({"frame": f, "R": R_cv.tolist(), "t": t_cv.tolist()})

#     # ── Export ────────────────────────────────────────────────────────────
#     data = {
#         "camera":        camera_name,
#         "start_frame":   start_frame,
#         "end_frame":     end_frame,
#         "render_width":  render_width,
#         "render_height": render_height,
#         "K":             K,
#         "frames":        frames,
#     }
#     with open(output_path, "w") as fh:
#         json.dump(data, fh, indent=2)

#     # Sanity check
#     R0 = np.array(frames[0]["R"])
#     t0 = np.array(frames[0]["t"])
#     print(f"✅ Exported {end_frame - start_frame + 1} frames → {output_path}")
#     print(f"   R row norms (should be [1,1,1]): {[round(float(np.linalg.norm(R0[r])),6) for r in range(3)]}")
#     print(f"   Frame-0 t   (should be ~[0,0,0]): {np.round(t0, 4).tolist()}")
#     print(f"   world_origin: {np.round(world_origin, 2).tolist()}")


# export_camera_data(
#     camera_name = "shot3:PF0003_04_Clip_4_REC709_2K_1000_1_1",
#     start_frame = 1010,
#     end_frame   = 1100,
#     output_path = "/Users/millyzhang/Desktop/project2/camera_data.json",
# )




# import json, numpy as np
# with open("/Users/millyzhang/Desktop/project/camera_data.json") as f:
#     d = json.load(f)
# print("world_origin:", d["world_origin"])
# print("frame 0 t:", d["frames"][0]["t"])
# print("frame 0 R row 2 (Z row):", d["frames"][0]["R"][2])



import maya.cmds as cmds
import json
import numpy as np


def export_camera_data(camera_name, start_frame, end_frame, output_path):
    cam_shape = cmds.listRelatives(camera_name, shapes=True)[0]

    # Intrinsics
    focal_length  = cmds.getAttr(f"{cam_shape}.focalLength")
    h_aperture    = cmds.getAttr(f"{cam_shape}.horizontalFilmAperture") * 25.4
    v_aperture    = cmds.getAttr(f"{cam_shape}.verticalFilmAperture")   * 25.4
    render_width  = cmds.getAttr("defaultResolution.width")
    render_height = cmds.getAttr("defaultResolution.height")
    fx = (focal_length / h_aperture) * render_width
    fy = (focal_length / v_aperture) * render_height
    cx = render_width  / 2.0
    cy = render_height / 2.0
    K  = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

    flip = np.diag([1.0, -1.0, -1.0])

    # collect all cam-to-world matrices
    R_c2w_list = []
    t_c2w_list = []

    for f in range(start_frame, end_frame + 1):
        cmds.currentTime(f)
        m = cmds.getAttr(f"{camera_name}.worldMatrix[0]")
        M = np.array(m).reshape(4, 4).T

        U, _, Vt = np.linalg.svd(M[:3, :3])
        R_c2w    = U @ Vt
        if np.linalg.det(R_c2w) < 0:
            U[:, -1] *= -1
            R_c2w = U @ Vt

        R_c2w_list.append(R_c2w)
        t_c2w_list.append(M[:3, 3])

    world_origin = t_c2w_list[0].copy()

    # compute world-to-camera extrinsics
    frames = []
    for i, f in enumerate(range(start_frame, end_frame + 1)):
        R_c2w   = R_c2w_list[i]
        cam_pos = t_c2w_list[i]

        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ cam_pos
        R_cv  = flip @ R_w2c
        t_cv  = flip @ t_w2c

        frames.append({"frame": f, "R": R_cv.tolist(), "t": t_cv.tolist()})

    # Export
    data = {
        "camera":        camera_name,
        "start_frame":   start_frame,
        "end_frame":     end_frame,
        "render_width":  render_width,
        "render_height": render_height,
        "K":             K,
        "frames":        frames,
    }
    with open(output_path, "w") as fh:
        json.dump(data, fh, indent=2)

    R0 = np.array(frames[0]["R"])
    t0 = np.array(frames[0]["t"])
    print(f"Exported {end_frame - start_frame + 1} frames → {output_path}")
    # print(f"R row norms (should be [1,1,1]): {[round(float(np.linalg.norm(R0[r])),6) for r in range(3)]}")
    # print(f"Frame-0 t   (should be ~[0,0,0]): {np.round(t0, 4).tolist()}")
    # print(f"world_origin: {np.round(world_origin, 2).tolist()}")


def run():
    # Get selected camera
    selection = cmds.ls(selection=True, long=True)
    camera = None
    for node in selection:
        if cmds.nodeType(node) == "camera":
            camera = cmds.listRelatives(node, parent=True, fullPath=True)[0]
            break
        shapes = cmds.listRelatives(node, shapes=True) or []
        if shapes and cmds.nodeType(shapes[0]) == "camera":
            camera = node
            break

    if not camera:
        print("Error: No camera selected. Please select a camera and run again.")
        return

    print(f"Camera: {camera}")

    # Prompt for inputs
    default_start = int(cmds.playbackOptions(q=True, minTime=True))
    default_end   = int(cmds.playbackOptions(q=True, maxTime=True))

    result = cmds.promptDialog(
        title="Start Frame",
        message=f"Start frame (timeline: {default_start}):",
        text=str(default_start),
        button=["OK", "Cancel"],
        defaultButton="OK",
        cancelButton="Cancel",
    )
    if result != "OK":
        print("Cancelled.")
        return
    start_frame = int(cmds.promptDialog(query=True, text=True))

    result = cmds.promptDialog(
        title="End Frame",
        message=f"End frame (timeline: {default_end}):",
        text=str(default_end),
        button=["OK", "Cancel"],
        defaultButton="OK",
        cancelButton="Cancel",
    )
    if result != "OK":
        print("Cancelled.")
        return
    end_frame = int(cmds.promptDialog(query=True, text=True))

    result = cmds.promptDialog(
        title="Output Path",
        message="Output path (.json):",
        text="/path/to/camera_data.json",
        button=["OK", "Cancel"],
        defaultButton="OK",
        cancelButton="Cancel",
    )
    if result != "OK":
        print("Cancelled.")
        return
    output_path = cmds.promptDialog(query=True, text=True).strip()

    # Validate
    if start_frame > end_frame:
        print(f"Error: Start frame ({start_frame}) must be ≤ end frame ({end_frame}).")
        return
    if not output_path.endswith(".json"):
        output_path += ".json"

    # Run
    export_camera_data(camera, start_frame, end_frame, output_path)


run()