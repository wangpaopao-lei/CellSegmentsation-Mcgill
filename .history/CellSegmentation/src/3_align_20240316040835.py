import numpy as np
import scipy.ndimage as ndi
import os
from typing import Tuple, Dict


def load_data(file_paths: Dict[str, str]) -> Dict[str, np.ndarray]:
    """
    Load data from text files.

    Parameters:
    file_paths (Dict[str, str]): A dictionary containing the file paths to load data from.

    Returns:
    Dict[str, np.ndarray]: A dictionary containing numpy arrays of the loaded data.
    """
    return {name: np.loadtxt(path) for name, path in file_paths.items()}


def process_probabilities(task1_result: np.ndarray, threshold: float) -> int:
    """
    Process probability data to calculate the number of points above a given threshold.

    Parameters:
    task1_result (np.ndarray): The cell nucleus probability data.
    threshold (float): The probability threshold.

    Returns:
    int: The number of points above the threshold.
    """
    probabilities = task1_result[:, 2]
    return np.sum(probabilities >= threshold)


def create_matrices(task1_result: np.ndarray, watershed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create matrices based on probability data and Watershed labels.
    Automatically determines the size of the matrices based on the maximum x and y coordinates.

    Parameters:
    task1_result (np.ndarray): The cell nucleus probability data.
    watershed (np.ndarray): The Watershed labels data.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the probability matrix and Watershed label matrix.
    """
    # Determine the size based on the max coordinates plus 1 (for 0-index)
    size_x = np.max(task1_result[:, 0]).astype(int) + 1
    size_y = np.max(task1_result[:, 1]).astype(int) + 1
    size = (size_x, size_y)

    task1_matrix = np.zeros(size)
    watershed_matrix = np.zeros(size)

    # Map probability values to the task1_matrix
    task1_matrix[task1_result[:, 0].astype(int), task1_result[:, 1].astype(int)] = task1_result[:, 2]

    # Map labels to the watershed_matrix
    watershed_matrix[watershed[:, 0].astype(int), watershed[:, 1].astype(int)] = watershed[:, 2]

    return task1_matrix, watershed_matrix




def label_and_compare(
    task1_matrix: np.ndarray,
    watershed_matrix: np.ndarray,
    threshold: float,
    min_cell_size: int,
) -> np.ndarray:
    """
    Label connected regions, compare with Watershed labels, and process new cell nuclei.

    Parameters:
    task1_matrix (np.ndarray): The probability matrix.
    watershed_matrix (np.ndarray): The Watershed label matrix.
    threshold (float): The probability threshold.
    min_cell_size (int): The minimum number of pixels for new cell nuclei.

    Returns:
    np.ndarray: The final label matrix.
    """
    task1_binary = task1_matrix >= threshold
    task1_labeled, _ = ndi.label(task1_binary)
    final_labels = np.zeros_like(task1_labeled)

    for label_num in np.unique(task1_labeled):
        if label_num == 0:
            continue  # Skip background
        mask = task1_labeled == label_num
        overlap = watershed_matrix[mask].astype(int)

        if overlap.size > 0 and np.any(overlap > 0):
            most_common = np.bincount(overlap[overlap > 0]).argmax()
        else:
            most_common = 0

        if most_common:
            final_labels[mask] = most_common
        elif np.sum(mask) > min_cell_size:
            distance = ndi.distance_transform_edt(~mask)
            index = np.unravel_index(np.argmin(distance), distance.shape)
            nearest_label = watershed_matrix[index]
            final_labels[mask] = nearest_label
    return final_labels


def save_results(
    task1_result: np.ndarray,
    final_labels: np.ndarray,
    filename: str = "results/task2_result.txt",
) -> None:
    """
    Save the final cell nucleus labels to a text file.

    Parameters:
    task1_result (np.ndarray): The original cell nucleus probability data.
    final_labels (np.ndarray): The final label matrix.
    filename (str, optional): The path to the result file. Defaults to "results/task2_result.txt".
    """
    final_labels_flat = final_labels[
        task1_result[:, :2].astype(int)[:, 0], task1_result[:, :2].astype(int)[:, 1]
    ]
    np.savetxt(
        filename, np.hstack((task1_result, final_labels_flat[:, None])), fmt="%f"
    )


def main() -> None:
    file_paths = {
        "task1_result": "dataset/task1_result.txt",
        "watershed": "dataset/watershed_labels.txt",
    }
    data = load_data(file_paths)
    threshold, min_cell_size = 0.01, 5

    num_above_threshold = process_probabilities(data["task1_result"], threshold)
    print(f"Points above threshold: {num_above_threshold}/{len(data['task1_result'])}")

    matrices = create_matrices(data["task1_result"], data["watershed"])
    final_labels = label_and_compare(*matrices, threshold, min_cell_size)

    save_results(data["task1_result"], final_labels)


if __name__ == "__main__":
    main()
