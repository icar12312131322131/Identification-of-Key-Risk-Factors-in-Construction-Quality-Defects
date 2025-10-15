import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# 设置全局字体为黑体（针对 Windows 系统）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径配置
INPUT_FILE_PATH = r"D:\reordered_tf.csv"
OUTPUT_DIR = r"D:\新建文件夹1"

def load_data(file_path):
    """读取CSV文件并将 't'、'f' 替换为布尔类型。"""
    df = pd.read_csv(file_path, index_col='text_id')
    df_bool = df.replace({'t': True, 'f': False})
    return df_bool

def generate_frequent_itemsets(data, min_support=0.6, max_len=2):
    """利用 Apriori 算法生成频繁项集。"""
    itemsets = apriori(data, min_support=min_support, use_colnames=True, max_len=max_len)
    return itemsets

def generate_association_rules(itemsets, min_confidence=0.8, min_lift=1.0):
    """生成关联规则。"""
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    # 筛选满足提升度阈值的规则
    rules = rules[rules['lift'] >= min_lift]
    return rules

def remove_frozenset(df):
    """去除 DataFrame 中 antecedents 和 consequents 列的 frozenset 符号"""
    df['antecedents'] = df['antecedents'].apply(lambda x: ', '.join(sorted(x)))
    df['consequents'] = df['consequents'].apply(lambda x: ', '.join(sorted(x)))
    return df

def save_results(itemsets, rules, output_dir):
    """保存频繁项集和关联规则到 CSV 文件。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frequent_itemsets_path = os.path.join(output_dir, "工作簿3.csv")
    association_rules_path = os.path.join(output_dir, "association_rules.csv")

    itemsets.to_csv(frequent_itemsets_path, index=False)
    rules = remove_frozenset(rules)
    rules.to_csv(association_rules_path, index=False)
    print("文件已成功保存。")

def visualize_results(itemsets, rules, output_dir):
    """生成数据的可视化图形。"""
    rules = remove_frozenset(rules)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # (1) 频繁项集支持度分布（前10个）
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='support',
        y='itemsets',
        data=itemsets.sort_values('support', ascending=False).head(10),
        palette='viridis'
    )
    plt.title("Top 10 频繁项集 (按支持度排序)")
    plt.xlabel("支持度")
    plt.ylabel("项集")
    plt.tight_layout()
    frequent_itemsets_png_path = os.path.join(output_dir, "frequent_itemsets.png")
    plt.savefig(frequent_itemsets_png_path, dpi=300)
    plt.show()

    # (2) 关联规则散点图（支持度 vs 置信度）
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=rules,
        x='support',
        y='confidence',
        size='lift',
        hue='lift',
        palette='viridis',
        sizes=(20, 200)
    )
    plt.title("关联规则散点图 (支持度 vs 置信度)")
    plt.xlabel("支持度")
    plt.ylabel("置信度")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    rules_scatter_png_path = os.path.join(output_dir, "rules_scatter.png")
    plt.savefig(rules_scatter_png_path, dpi=300)
    plt.show()

    # (3) 关联规则网络图：节点大小表示 PageRank 重要性，边宽表示提升度，移除边上文本
    try:
        import networkx as nx
        plt.figure(figsize=(12, 8))
        # 构建图时，注意规则中的 antecedents 和 consequents 已经转换为字符串
        G = nx.from_pandas_edgelist(
            rules,
            source='antecedents',
            target='consequents',
            edge_attr=['lift', 'confidence']
        )
        
        # 计算节点的 PageRank
        pagerank = nx.pagerank(G)
        # 根据 PageRank 值设置节点大小，这里可以乘以一个系数调整视觉效果
        node_sizes = [v * 3000 for v in pagerank.values()]
        
        # 节点布局
        pos = nx.spring_layout(G, k=0.5)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_sizes)
        # 边宽根据提升度设置
        edges = G.edges(data=True)
        edge_widths = [attr['lift'] * 0.5 for _, _, attr in edges]
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=edge_widths)
        nx.draw_networkx_labels(G, pos, font_family='SimHei', font_size=10)

        plt.title("关联规则网络图 (节点大小=PageRank重要性，边宽=提升度)")
        plt.axis('off')
        rules_network_png_path = os.path.join(output_dir, "rules_network.png")
        plt.savefig(rules_network_png_path, dpi=300, bbox_inches='tight')
        plt.show()
    except ImportError:
        print("未安装 networkx，跳过网络图生成。安装命令：pip install networkx")

def main():
    start_time = time.time()

    try:
        # 1. 数据加载和预处理
        data_bool = load_data(INPUT_FILE_PATH)
        print(f"数据加载耗时: {time.time() - start_time:.2f}s")
    except FileNotFoundError:
        print(f"Error: 文件 {INPUT_FILE_PATH} 未找到。")
        return
    except Exception as e:
        print(f"数据加载时发生错误: {e}")
        return

    try:
        # 2. 生成频繁项集及关联规则
        frequent_itemsets = generate_frequent_itemsets(data_bool, min_support=0.12, max_len=2)
        print(f"Apriori算法耗时: {time.time() - start_time:.2f}s")

        rules = generate_association_rules(frequent_itemsets, min_confidence=0.6, min_lift=1.0)

        # 保存结果
        save_results(frequent_itemsets, rules, OUTPUT_DIR)
    except Exception as e:
        print(f"Apriori 或规则生成过程中发生错误: {e}")
        return

    # 3. 可视化分析
    visualize_results(frequent_itemsets, rules, OUTPUT_DIR)
    print(f"总耗时: {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    main()
