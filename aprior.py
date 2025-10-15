import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# Set global font for Windows (optional, adjust if needed)
plt.rcParams['font.sans-serif'] = ['Arial']  # changed from SimHei for English
plt.rcParams['axes.unicode_minus'] = False

# File path configuration
INPUT_FILE_PATH = r"D:\reordered_tf.csv"
OUTPUT_DIR = r"D:\results"

def load_data(file_path):
    """Load CSV data and convert 't'/'f' to boolean values."""
    df = pd.read_csv(file_path, index_col='text_id')
    df_bool = df.replace({'t': True, 'f': False})
    return df_bool

def generate_frequent_itemsets(data, min_support=0.6, max_len=2):
    """Generate frequent itemsets using the Apriori algorithm."""
    itemsets = apriori(data, min_support=min_support, use_colnames=True, max_len=max_len)
    return itemsets

def generate_association_rules(itemsets, min_confidence=0.8, min_lift=1.0):
    """Generate association rules and filter by lift threshold."""
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules['lift'] >= min_lift]
    return rules

def remove_frozenset(df):
    """Convert antecedents and consequents from frozenset to comma-separated strings."""
    df['antecedents'] = df['antecedents'].apply(lambda x: ', '.join(sorted(x)))
    df['consequents'] = df['consequents'].apply(lambda x: ', '.join(sorted(x)))
    return df

def save_results(itemsets, rules, output_dir):
    """Save frequent itemsets and association rules as CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    frequent_itemsets_path = os.path.join(output_dir, "frequent_itemsets.csv")
    association_rules_path = os.path.join(output_dir, "association_rules.csv")

    itemsets.to_csv(frequent_itemsets_path, index=False)
    rules = remove_frozenset(rules)
    rules.to_csv(association_rules_path, index=False)
    print("Results successfully saved.")

def visualize_results(itemsets, rules, output_dir):
    """Visualize frequent itemsets and association rules."""
    rules = remove_frozenset(rules)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Top 10 frequent itemsets by support
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='support',
        y='itemsets',
        data=itemsets.sort_values('support', ascending=False).head(10),
        palette='viridis'
    )
    plt.title("Top 10 Frequent Itemsets (by Support)")
    plt.xlabel("Support")
    plt.ylabel("Itemsets")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "frequent_itemsets.png"), dpi=300)
    plt.show()

    # 2. Association rules scatter plot (Support vs Confidence)
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
    plt.title("Association Rules Scatter Plot (Support vs Confidence)")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rules_scatter.png"), dpi=300)
    plt.show()

    # 3. Network graph of association rules (node size = PageRank, edge width = lift)
    try:
        import networkx as nx
        plt.figure(figsize=(12, 8))
        G = nx.from_pandas_edgelist(
            rules,
            source='antecedents',
            target='consequents',
            edge_attr=['lift', 'confidence']
        )
        pagerank = nx.pagerank(G)
        node_sizes = [v * 3000 for v in pagerank.values()]
        pos = nx.spring_layout(G, k=0.5)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_sizes)
        edge_widths = [attr['lift'] * 0.5 for _, _, attr in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=edge_widths)
        nx.draw_networkx_labels(G, pos, font_size=10)

        plt.title("Association Rules Network (Node size=PageRank, Edge width=Lift)")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "rules_network.png"), dpi=300, bbox_inches='tight')
        plt.show()
    except ImportError:
        print("networkx not installed, skipping network graph. Install via `pip install networkx`")

def main():
    start_time = time.time()

    try:
        # 1. Load and preprocess data
        data_bool = load_data(INPUT_FILE_PATH)
        print(f"Data loading time: {time.time() - start_time:.2f}s")
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE_PATH} not found.")
        return
    except Exception as e:
        print(f"Error during data loading: {e}")
        return

    try:
        # 2. Generate frequent itemsets and association rules
        frequent_itemsets = generate_frequent_itemsets(data_bool, min_support=0.12, max_len=2)
        print(f"Apriori execution time: {time.time() - start_time:.2f}s")

        rules = generate_association_rules(frequent_itemsets, min_confidence=0.6, min_lift=1.0)

        save_results(frequent_itemsets, rules, OUTPUT_DIR)
    except Exception as e:
        print(f"Error during Apriori or rules generation: {e}")
        return

    # 3. Visualization
    visualize_results(frequent_itemsets, rules, OUTPUT_DIR)
    print(f"Total execution time: {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    main()
