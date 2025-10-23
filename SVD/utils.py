from scipy.spatial.transform import Rotation
import ezc3d
from scipy.signal import butter, filtfilt
import numpy as np

def load_marker_data(c3d_path):
    """
    Load marker positions from C3D file.
    
    Args:
        c3d_path: Path to the C3D file
        
    Returns:
        marker_positions: Array of shape (framecount, num_markers, 3) - marker positions over time
    """
    c3d = ezc3d.c3d(c3d_path)
    
    # Extract points data: shape (4, num_markers, framecount)
    points = c3d['data']['points']
    marker_positions = points[:3, :, :]  # (3, num_markers, framecount) ignore the 4th dimension (all 1's)

    # Transpose to get (framecount, num_markers, 3)
    marker_positions = marker_positions.transpose(2, 1, 0)
    
    return marker_positions / 1000  # Convert from mm to meters


def lowpass_filter(data: np.ndarray, cutoff_freq: float, fs: float) -> np.ndarray:
    nyquist = 0.5 * fs
    norm_cutoff = cutoff_freq / nyquist
    b, a = butter(N=4, Wn=norm_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def estimate_rigid_body_pose(markers):
    """
    Estimate the rigid-body pose (centroid + orientation) from marker positions over time,
    using only anatomical frame definitions (no SVD).

    Args:
        markers: Array of shape (n_frames, n_markers, 3) with marker positions

    Returns:
        centroids: Array of shape (n_frames, 3) with centroid positions
        rotation_matrices: Array of shape (n_frames, 3, 3) with rotation matrices
    """
    n_frames = markers.shape[0]
    centroids = np.nanmean(markers, axis=1)
    rotation_matrices = np.empty((n_frames, 3, 3))
    
    # Assuming markers[:, 0] is L20, markers[:, 1] is R20, markers[:, 2] is 11, markers[:, 3] is 12
    for frame in range(n_frames):
        # Define anatomical coordinate system directly from markers
        forward_vec = markers[frame, 2] - markers[frame, 3]  # back to front vector
        side_vec = markers[frame, 0] - markers[frame, 1]     # right to left vector
        
        # Create orthogonal coordinate system
        x_axis = forward_vec / np.linalg.norm(forward_vec)
        temp_z = np.cross(forward_vec, side_vec)
        z_axis = temp_z / np.linalg.norm(temp_z)
        y_axis = np.cross(z_axis, x_axis)
        
        # Store rotation matrix (columns are the axes of the anatomical frame)
        rotation_matrices[frame] = np.column_stack((x_axis, y_axis, z_axis))
    
    return centroids, rotation_matrices


def obtain_3_rank_trajectory(saddle_centers, rotation_matrices):
    fs = 240.0  # Sampling frequency in Hz
    n_frames = saddle_centers.shape[0]

    # Calculate centroid linear velocities in global frame first
    global_linear_velocities = np.gradient(saddle_centers, 1.0/fs, axis=0)
    # Initialize arrays for local frame velocities
    local_linear_velocities = np.zeros_like(global_linear_velocities)
    local_angular_velocities = np.zeros((n_frames, 3))

    # For each frame
    for i in range(n_frames):
        # Transform linear velocity from global to local frame
        # R.T converts from global to local coordinates
        local_linear_velocities[i] = rotation_matrices[i].T @ global_linear_velocities[i]
        
        # Calculate angular velocities in local frame
        if i > 0:
            R_prev = rotation_matrices[i-1]
            R_curr = rotation_matrices[i]
            
            # Calculate relative rotation (same as before)
            rel_rotation = Rotation.from_matrix(R_curr @ R_prev.T)
            
            # Get rotation vector (global frame)
            global_rot_vec = rel_rotation.as_rotvec()
            
            # Transform to local frame using the previous frame's orientation
            local_rot_vec = R_prev.T @ global_rot_vec
            
            # Scale by 1/dt to get angular velocity (rad/s)
            local_angular_velocities[i] = local_rot_vec * fs

    # First frame angular velocity
    local_angular_velocities[0] = local_angular_velocities[1]

    # Combine local frame velocities into matrix A
    A = np.hstack((local_linear_velocities, local_angular_velocities))
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    A_k = (U[:, :3] * S[:3]) @ Vt[:3, :]
    
    # Extract local linear and angular velocities from the low-rank approximation
    local_linear_vel_k = A_k[:, :3]  # (n_frames, 3)
    local_angular_vel_k = A_k[:, 3:] # (n_frames, 3)

    # Initialize arrays for global velocities
    global_linear_vel_k = np.zeros_like(local_linear_vel_k)

    # Transform local linear velocities to global frame
    for i in range(n_frames):
        # R converts from local to global coordinates
        global_linear_vel_k[i] = rotation_matrices[i] @ local_linear_vel_k[i]

    # Now let's integrate to get positions and orientations
    # Initialize arrays for the reconstructed trajectory
    pos_reconstructed = np.zeros((n_frames, 3))
    rot_reconstructed = np.zeros((n_frames, 3, 3))

    # Set initial position and orientation
    pos_reconstructed[0] = saddle_centers[0]
    rot_reconstructed[0] = rotation_matrices[0]

    # Euler integration
    dt = 1.0/fs  # Time step

    for i in range(1, n_frames):
        # Update position using global linear velocity
        pos_reconstructed[i] = pos_reconstructed[i-1] + global_linear_vel_k[i-1] * dt
        
        # Update orientation using local angular velocity
        # Convert angular velocity to rotation vector
        rot_vec = local_angular_vel_k[i-1] * dt
        
        # Create a rotation object from rotation vector
        rel_rot = Rotation.from_rotvec(rot_vec)
        
        # Previous rotation as Rotation object
        prev_rot = Rotation.from_matrix(rot_reconstructed[i-1])
        
        # Apply relative rotation (in local frame)
        new_rot = prev_rot * rel_rot
        
        # Store the new rotation matrix
        rot_reconstructed[i] = new_rot.as_matrix()
    return pos_reconstructed, rot_reconstructed