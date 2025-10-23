import mujoco
import mujoco.viewer
import time
from utils import *

def setup_mujoco_scene():
    """
    Create a MuJoCo scene with a floor and lighting.
    
    Returns:
        model: MuJoCo model
        data: MuJoCo data
    """
    # Create a simple XML scene with floor and lighting
    xml_string = """
    <mujoco model="visualization">
        <worldbody>
            <light directional="false" pos="0 0 17" dir="0 0 -1" castshadow="true"/>
            <geom name="floor" size="16 20 .125" pos="0 0 -0.2" type="plane"/>
        </worldbody>

        <visual>
            <rgba haze=".15 .25 .35 1"/>
        </visual>
    </mujoco>
    """
    
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    
    return model, data


def draw_markers(viewer, marker_positions):
    """
    Draw markers as red spheres in the MuJoCo viewer.
    """
    for marker_idx, pos in enumerate(marker_positions):
        # Add a new geometry to the scene if we have space
        if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
            # Get the next available geometry slot
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            color = np.array([1.0, 0.0, 0.0, 1.0])  # Default Red color
            # Initialize the sphere
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.02, 0, 0]),  # Sphere radius
                pos=pos,
                mat=np.identity(3).flatten(),
                rgba=color
            )
            
            # Increment the geometry count
            viewer.user_scn.ngeom += 1
    
    return len(marker_positions) # Return the number of markers drawn

def draw_markers_from_list(viewer, marker_list, color=np.array([0.0, 0.5, 1.0, 1.0]), size=.03):
    for com in marker_list:
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([size, 0, 0]),  # Sphere radius
                    pos=com,
                    mat=np.identity(3).flatten(),
                    rgba=color
                )
        viewer.user_scn.ngeom += 1

def draw_coordinate_frame(viewer, center, rotation_matrix, frame_size=0.5, color_variant=0):
    # Define a list of distinct colors (RGBA format)
    color_palette = [
        np.array([1.0, 0.0, 0.0, 1.0]),  # Red
        np.array([1.0, 0.0, 1.0, 1.0]),  # Magenta
        np.array([1.0, 0.5, 0.0, 1.0]),  # Orange
        np.array([0.0, 1.0, 0.0, 1.0]),  # Green
        np.array([1.0, 1.0, 0.0, 1.0]),  # Yellow
        np.array([0.5, 1.0, 0.0, 1.0]),  # Lime
        np.array([0.0, 0.0, 1.0, 1.0]),  # Blue
        np.array([0.0, 1.0, 1.0, 1.0]),  # Cyan
        np.array([0.5, 0.0, 0.5, 1.0]),  # Purple
    ]
    
    # Select three colors from the palette based on the color_variant
    axis_colors = [
        color_palette[(color_variant) % len(color_palette)],
        color_palette[(color_variant + 3) % len(color_palette)],  # Skip ahead for contrast
        color_palette[(color_variant + 6) % len(color_palette)]   # Skip ahead further
    ]
    
    # Draw each axis
    if viewer.user_scn.ngeom + 3 < viewer.user_scn.maxgeom:
        for i in range(3):
            # Extract axis direction from rotation matrix
            axis = rotation_matrix[:, i]
            
            # Create orientation matrix where z-axis aligns with our direction
            z_axis = axis / np.linalg.norm(axis)
            
            # Find perpendicular vectors to form a basis
            if abs(z_axis[0]) < abs(z_axis[1]):
                x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
            else:
                x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
                
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            # Create rotation matrix
            orientation = np.column_stack((x_axis, y_axis, z_axis))
            
            # Draw axis as arrow
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.array([frame_size/30, frame_size/20, frame_size]), # shaft radius, head radius, length
                pos=center,  # Start at the center
                mat=orientation.flatten(),
                rgba=axis_colors[i]
            )
            viewer.user_scn.ngeom += 1

def play_animation(marker_positions, saddle_indices, frame_rate=240):
    """
    Play animation of markers and ground reaction forces.
    
    Args:
        marker_positions: Array of marker positions (framecount, 99, 3)
        frame_rate: Animation frame rate (default 240 Hz)
    """
    # Setup MuJoCo scene
    model, data = setup_mujoco_scene()
    
    # Calculate time step
    dt = 1.0 / frame_rate
    
    # Start the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0
        max_frames = marker_positions.shape[0]

        # Configure camera position and orientation
        viewer.cam.lookat[0] = 0.0  # x position to look at
        viewer.cam.lookat[1] = 0.0  # y position to look at
        viewer.cam.lookat[2] = 0.0  # z position to look at (floor level)
        viewer.cam.distance = 15.0   # Distance from lookat point
        viewer.cam.elevation = -20  # Camera elevation angle (degrees)
        viewer.cam.azimuth = 45     # Camera azimuth angle (degrees)

        # compute centroid of the saddle markers (traj shape: (n_frames, 4, 3))
        saddle_centers, rotation_matrices = estimate_rigid_body_pose(marker_positions[:, saddle_indices, :])
        pos_reconstructed, rot_reconstructed = obtain_3_rank_trajectory(saddle_centers, rotation_matrices)
        while viewer.is_running():
            step_start = time.time()
            # Clear previous geometries
            viewer.user_scn.ngeom = 0

            # Draw current frame
            draw_markers(viewer, marker_positions[frame])
            #print(marker_positions[frame][markers_for_saddle])

            ### Saddle Position coloring ###
            # Draw the markers used for saddle position estimation in orange
            draw_markers_from_list(viewer, marker_positions[frame][saddle_indices], color=np.array([1.0, 0.5, 0.0, 1.0]))
            draw_coordinate_frame(viewer, saddle_centers[frame], rotation_matrices[frame])
            draw_coordinate_frame(viewer, pos_reconstructed[frame], rot_reconstructed[frame], color_variant=1)

            # Update viewer
            mujoco.mj_step(model, data)
            viewer.sync()

            # Advance frame, ensures loop
            frame = (frame + 1) % max_frames
            
            # Rudimentary time keeping, will drift relative to wall clock
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def main():
    """
    Main function to load data and start visualization.
    """
    c3d_path = "/home/darren/Desktop/hippobotamus/SVD/C3D_DATA/20201128_ID_2_0008.c3d"
    marker_clouds = load_marker_data(c3d_path)
    marker_clouds = lowpass_filter(marker_clouds, cutoff_freq=5, fs=240)

    c3d = ezc3d.c3d(c3d_path)
    labels = c3d['parameters']['POINT']['LABELS']['value']

    def to_str(x):
        return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
    saddle_indices = np.array([i for i, lab in enumerate(labels) if to_str(lab).strip().endswith(('11', '12', '20'))])

    # Start animation
    play_animation(marker_clouds, saddle_indices)

if __name__ == "__main__":
    main()