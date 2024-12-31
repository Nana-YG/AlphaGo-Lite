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
        # Iterate through datasets in the file
        for dataset_name in h5_file.keys():
            try:
                board = h5_file[f"{dataset_name}"][:]
                liberty = h5_file[f"liberty_{dataset_name.split('_')[1]}"][:]

                # Generate additional matrices
                ones_matrix = np.ones_like(board, dtype=np.float32)
                zeros_matrix = np.zeros_like(board, dtype=np.float32)

                # Combine into a 4-channel array
                combined_array = np.stack([board, liberty, ones_matrix, zeros_matrix], axis=0)

                # Save as .npy file
                output_file_path = os.path.join(output_dir, f"{dataset_name}.npy")
                np.save(output_file_path, combined_array)

                print(f"Saved: {output_file_path}")
            except KeyError as e:
                print(f"Missing corresponding dataset for {dataset_name}: {e}")
            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {e}")

# Example usage
h5_file_path = "visualTest.h5"
output_dir = "npy-images/"
convert_h5_to_npy(h5_file_path, output_dir)
