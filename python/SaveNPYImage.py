import os
import h5py
import numpy as np

def convert_h5_to_npy(h5_file_path, output_dir):
    """
    Converts data from an .h5 file to .npy format.

    Args:
        h5_file_path (str): Path to the input .h5 file.
        output_dir (str): Directory to save the .npy files.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_file_path, 'r') as h5_file:
        # Iterate through groups in the file
        for group_name in h5_file.keys():
            group = h5_file[group_name]

            for i in range(len(group.keys()) // 3):  # Assuming each group has `board_`, `liberty_`, and `nextMove_`
                try:
                    # Load board and liberty data
                    board = group[f"board_{i}"][:]
                    liberty = group[f"liberty_{i}"][:]

                    # Generate additional matrices
                    ones_matrix = np.ones_like(board, dtype=np.float32)
                    zeros_matrix = np.zeros_like(board, dtype=np.float32)

                    # Combine into a 4-channel array
                    combined_array = np.stack([board, liberty, ones_matrix, zeros_matrix], axis=0)

                    # Save as .npy file
                    output_file_path = os.path.join(output_dir, f"{group_name}_{i}.npy")
                    np.save(output_file_path, combined_array)

                    print(f"Saved: {output_file_path}")
                except KeyError as e:
                    print(f"Missing data for index {i} in group {group_name}: {e}")
                except Exception as e:
                    print(f"Error processing group {group_name}, index {i}: {e}")

# Example usage
h5_file_path = "visualTest.h5"
output_dir = "npy-images/"
convert_h5_to_npy(h5_file_path, output_dir)
