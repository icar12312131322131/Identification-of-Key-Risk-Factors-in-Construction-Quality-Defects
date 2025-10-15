import pandas as pd
import numpy as np

def read_adjacency_matrix(file_path):
    """
    Read an adjacency matrix from an Excel file.
    Assumes the file contains a square matrix with no row or column headers.
    """
    df = pd.read_excel(file_path, header=None)
    matrix = df.values.astype(int)
    return matrix


def compute_reachability(matrix):
    """
    Compute the reachability (transitive closure) matrix using the Warshall algorithm.
    Adds reflexivity before iterative updates.
    """
    n = matrix.shape[0]
    R = (matrix.astype(bool)).copy()
    for i in range(n):
        R[i, i] = True

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if not R[i, j]:
                    R[i, j] = R[i, k] and R[k, j]

    return R.astype(int)


def ism_level_partitioning(R):
    """
    Perform ISM level partitioning based on the reachability matrix R.
    For each remaining element i, define:
        - Reachable set: {j in remaining | R[i][j] == 1}
        - Antecedent set: {j in remaining | R[j][i] == 1}
    If the reachable set of i is a subset of its antecedent set,
    then i belongs to the current level.
    """
    n = R.shape[0]
    all_indices = set(range(n))
    levels = {}
    current_level = 1
    remaining = all_indices.copy()

    while remaining:
        current_level_nodes = []
        for i in remaining:
            reachable = {j for j in remaining if R[i, j] == 1}
            antecedent = {j for j in remaining if R[j, i] == 1}
            if reachable.issubset(antecedent):
                current_level_nodes.append(i)

        if not current_level_nodes:
            # Prevent infinite loops if no node satisfies the condition
            current_level_nodes = list(remaining)

        levels[current_level] = current_level_nodes
        remaining = remaining - set(current_level_nodes)
        current_level += 1

    return levels


def main():
    # Path to the Excel file containing the adjacency matrix
    file_path = r"D:\adjacency_matrix.xlsx"

    # 1. Read the adjacency matrix
    matrix = read_adjacency_matrix(file_path)
    print("Adjacency Matrix:")
    print(matrix)

    # 2. Compute the reachability matrix
    R = compute_reachability(matrix)
    print("\nReachability Matrix:")
    print(R)

    # 3. Perform ISM level partitioning
    levels = ism_level_partitioning(R)
    print("\nISM Level Partitioning Results:")
    for level, nodes in levels.items():
        print(f"Level {level}: Elements {nodes}")


if __name__ == "__main__":
    main()
