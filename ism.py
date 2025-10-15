import pandas as pd
import numpy as np

def read_adjacency_matrix(file_path):
    """
    读取 Excel 文件中的邻接矩阵。
    假设文件中存储的是一个方阵，且没有行列标题。
    """
    df = pd.read_excel(file_path, header=None)
    matrix = df.values.astype(int)
    return matrix

def compute_reachability(matrix):
    """
    利用 Warshall 算法计算传递闭包（reachability matrix）。
    先增加自反性，再迭代更新可达关系。
    """
    n = matrix.shape[0]
    # 将矩阵转换为布尔类型，并加入自反性
    R = (matrix.astype(bool)).copy()
    for i in range(n):
        R[i, i] = True

    # Warshall 算法计算传递闭包
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if not R[i, j]:
                    R[i, j] = R[i, k] and R[k, j]
    # 转换为整数形式，便于后续使用（1 表示可达，0 表示不可达）
    return R.astype(int)

def ism_level_partitioning(R):
    """
    根据传递闭包 R，对指标进行层次划分。
    对于剩余集合中的每个指标 i，计算：
      - 可达集：{j in remaining | R[i][j]==1}
      - 先行集：{j in remaining | R[j][i]==1}
    若可达集是先行集的子集（即 R_i ∩ A_i == R_i），则 i 属于当前层。
    """
    n = R.shape[0]
    all_indices = set(range(n))
    levels = {}  # key: 层级，value: 指标列表（指标编号）
    current_level = 1
    remaining = all_indices.copy()

    while remaining:
        current_level_nodes = []
        for i in remaining:
            # 仅考虑剩余指标中的可达关系
            reachable = {j for j in remaining if R[i, j] == 1}
            antecedent = {j for j in remaining if R[j, i] == 1}
            # 如果指标 i 的可达集在先行集中，则 i 属于当前层
            if reachable.issubset(antecedent):
                current_level_nodes.append(i)
        if not current_level_nodes:
            # 防止死循环：如果没有找到满足条件的指标，则将剩余所有指标归为同一层
            current_level_nodes = list(remaining)
        levels[current_level] = current_level_nodes
        # 从剩余指标中移除当前层指标
        remaining = remaining - set(current_level_nodes)
        current_level += 1

    return levels

def main():
    # 设置 Excel 文件路径（请根据实际情况修改路径）
    file_path = r"D:\邻接.xlsx"

    # 1. 读取邻接矩阵
    matrix = read_adjacency_matrix(file_path)
    print("原始邻接矩阵：")
    print(matrix)

    # 2. 计算传递闭包
    R = compute_reachability(matrix)
    print("\n传递闭包（Reachability Matrix）：")
    print(R)

    # 3. 进行 ISM 层次划分
    levels = ism_level_partitioning(R)
    print("\nISM 层次划分结果：")
    for level, nodes in levels.items():
        print(f"第 {level} 层：指标 {nodes}")

if __name__ == "__main__":
    main()
