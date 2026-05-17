import maya.cmds as cmds
import numpy as np
import os

# User input for file paths and start frame
# Prompt for .npy path
result = cmds.promptDialog(
    title="Pose Data",
    message="Enter path to predicted_3d_world.npy file:",
    text="",
    button=["OK", "Cancel"],
    defaultButton="OK",
    cancelButton="Cancel",
)
if result != "OK":
    print("Cancelled.")
    raise SystemExit
POSE3D_NPY = cmds.promptDialog(query=True, text=True).strip()
 
# Prompt for start frame
default_frame = int(cmds.currentTime(q=True))
result = cmds.promptDialog(
    title="Start Frame",
    message="Enter start frame:",
    text=str(default_frame),
    button=["OK", "Cancel"],
    defaultButton="OK",
    cancelButton="Cancel",
)
if result != "OK":
    print("Cancelled.")
    raise SystemExit
raw = cmds.promptDialog(query=True, text=True).strip()
START_FRAME = int(raw) if raw.lstrip("-").isdigit() else default_frame

# Joint order in the .npy file (VideoPose3D format)
JOINTS_ORDERED = [
    "Hip",
    "RightHip","RightKnee","RightAnkle",
    "LeftHip","LeftKnee","LeftAnkle",
    "Spine","Thorax","Neck","Head",
    "LeftShoulder","LeftElbow","LeftWrist",
    "RightShoulder","RightElbow","RightWrist",
]
# Parent map for building skeleton hierarchy
PARENT_MAP = {
    "Hip": None,
    "RightHip":"Hip","RightKnee":"RightHip","RightAnkle":"RightKnee",
    "LeftHip":"Hip","LeftKnee":"LeftHip","LeftAnkle":"LeftKnee",
    "Spine":"Hip","Thorax":"Spine","Neck":"Thorax","Head":"Neck",
    "LeftShoulder":"Thorax","LeftElbow":"LeftShoulder","LeftWrist":"LeftElbow",
    "RightShoulder":"Thorax","RightElbow":"RightShoulder","RightWrist":"RightElbow",
}

# Load 3D pose data
poses = np.load(POSE3D_NPY)
F = poses.shape[0]
J = poses.shape[1]
print(f"Loaded: {F} frames")

# Locators
def create_locators():
    locs = {}
    for j in JOINTS_ORDERED:
        name = f"LOC_{j}"
        if cmds.objExists(name):
            cmds.delete(name)
        locs[j] = cmds.spaceLocator(name=name)[0]
    return locs

def animate_locators(locs):
    for f in range(F):
        cmds.currentTime(START_FRAME + f)
        for i, j in enumerate(JOINTS_ORDERED):
            cmds.xform(
                locs[j],
                ws=True,
                t=poses[f, i].tolist()
            )
            cmds.setKeyframe(locs[j], attribute="translate")

# Skeleton
def build_skeleton():
    joints = {}
    for j in JOINTS_ORDERED:
        name = f"predicted_{j}"
        if cmds.objExists(name):
            cmds.delete(name)

        cmds.select(clear=True)
        joints[j] = cmds.joint(name=name, p=(0,0,0))

    # parenting pass
    for child, parent in PARENT_MAP.items():
        if parent:
            cmds.parent(joints[child], joints[parent])

    print("Skeleton built")
    return joints

# Constrains (locators to joints)
def constrain(locs, joints):
    for j in JOINTS_ORDERED:
        cmds.parentConstraint(
            locs[j],
            joints[j],
            mo=False
        )
    print("Constraints applied")

# Bake results to joints
def bake(joints):
    cmds.bakeResults(
        list(joints.values()),
        simulation=True,
        time=(START_FRAME, START_FRAME + F - 1),
        sampleBy=1,
        disableImplicitControl=True,
        preserveOutsideKeys=True
    )
    print("Baked animation")


def run():
    print("=== START ===")

    locs = create_locators()
    animate_locators(locs)

    joints = build_skeleton()
    constrain(locs, joints)

    bake(joints)

    print("=== DONE ===")


run()