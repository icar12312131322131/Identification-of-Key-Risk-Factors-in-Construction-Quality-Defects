import sys
import re
import jieba
import pandas as pd
import numpy as np
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import networkx as nx  # 用于构造图并计算TextRank

# 配置参数
CONFIG = {
    "data_path": "D:\\shiyan.xlsx",
    "sheet_name": "问题和原因",
    "column_name": "A",
    "stopwords_path": "D:\\文本挖掘\\hit_stopwords.txt",
    "special_words_path": "D:\\文本挖掘\\zhuanyongci.txt",
    "output_keywords_num": 150,
    "ngram_range": (1, 1),
    "max_candidates_per_doc": 50,  # 每个文档候选关键词最大数量
    "similarity_threshold": 0.5,   # 候选词与文档的最小余弦相似度
    "min_doc_coverage": 0.05,      # 最低文档覆盖率
    "clustering_num":  5           # 聚类簇数
}

# 初始化语义模型
sentence_model = SentenceTransformer("moka-ai/m3e-base")

def load_dictionaries():
    """
    加载停用词和专用词，并将专用词加入jieba词典
    """
    with open(CONFIG['stopwords_path'], 'r', encoding='utf-8') as f:
        stopwords = set(f.read().splitlines())
    with open(CONFIG['special_words_path'], 'r', encoding='utf-8') as f:
        special_words = set(f.read().splitlines())
        for word in special_words:
            jieba.add_word(word)
    return stopwords, special_words

def enhanced_preprocess(text, stopwords):
    """
    增强型文本预处理：去除标点、编号和停用词
    """
    text = re.sub(r'[^\w\s]', '', str(text))         # 去除标点
    text = re.sub(r'^\d+[.、]', '', text)              # 去除开头的编号
    words = jieba.lcut(text)
    filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
    return ' '.join(filtered_words)

def dynamic_weight_adjustment(keyword):
    """
    原有动态权重调整（本次不再使用，可保留备用）
    """
    length = len(keyword)
    if length >= 3:
        return 1.2
    elif length == 2:
        return 1.0
    return 0.8

def cross_validate_filter(candidates, doc_embeddings):
    """
    利用候选词与所有文档的余弦相似度进行交叉验证过滤，
    仅保留与至少一个文档相似度高于阈值的候选词。
    """
    candidate_embeddings = sentence_model.encode(candidates)
    doc_matrix = np.array(doc_embeddings)
    max_similarities = []
    for cand_emb in candidate_embeddings:
        similarities = cosine_similarity([cand_emb], doc_matrix)[0]
        max_similarities.append(np.max(similarities))
    return [candidates[i] for i, s in enumerate(max_similarities) if s > CONFIG['similarity_threshold']]

def process_single_document(text, stopwords, kw_model):
    """
    处理单个文档：预处理、候选关键词提取和文档向量生成
    """
    processed = enhanced_preprocess(text, stopwords)
    if not processed.strip():
        return [], None
    # 使用KeyBERT提取候选关键词
    keywords = kw_model.extract_keywords(
        processed,
        keyphrase_ngram_range=CONFIG['ngram_range'],
        stop_words=None,
        top_n=CONFIG['max_candidates_per_doc'],
        diversity=0.7
    )
    valid_keywords = [kw for kw, score in keywords if score > 0.3]
    doc_embedding = sentence_model.encode(processed)
    return valid_keywords, doc_embedding

def cluster_keywords_textrank(candidates, embeddings):
    """
    对候选关键词进行KMeans聚类，然后在每个簇内基于候选关键词的余弦相似度构造图，
    利用PageRank（TextRank）算法计算每个关键词的得分，最终返回聚类结果：
    返回值 cluster_results 为字典： {簇编号: [(关键词, TextRank得分), ...]}，同时返回各候选词的聚类标签
    """
    if not candidates:
        return {}, []
    embeddings = np.array(embeddings)
    n_clusters = min(CONFIG['clustering_num'], len(candidates))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    cluster_results = {}
    # 针对每个簇独立计算TextRank
    for cluster in np.unique(cluster_labels):
        indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
        cluster_keywords = [candidates[i] for i in indices]
        cluster_embeddings = embeddings[indices]
        # 计算簇内关键词的相似度矩阵
        sim_matrix = cosine_similarity(cluster_embeddings)
        # 构造图：节点为簇内每个关键词，边权为关键词之间的相似度（仅保留相似度大于0的边）
        graph = nx.Graph()
        for i, kw in enumerate(cluster_keywords):
            graph.add_node(i, keyword=kw)
        for i in range(len(cluster_keywords)):
            for j in range(i+1, len(cluster_keywords)):
                sim = sim_matrix[i, j]
                if sim > 0:
                    graph.add_edge(i, j, weight=sim)
        # 若簇内只有一个关键词，则直接赋予得分1.0，否则利用PageRank计算TextRank得分
        if len(cluster_keywords) == 1:
            scores = {0: 1.0}
        else:
            scores = nx.pagerank(graph, weight='weight')
        # 将得分映射到关键词
        cluster_keyword_scores = [(cluster_keywords[i], scores.get(i, 0)) for i in range(len(cluster_keywords))]
        # 按得分降序排序
        cluster_keyword_scores.sort(key=lambda x: x[1], reverse=True)
        cluster_results[cluster] = cluster_keyword_scores
    return cluster_results, cluster_labels

def extract_global_keywords(texts, stopwords):
    """
    提取全局关键词：
    1. 针对每个文档预处理后使用KeyBERT提取候选词和生成文档向量
    2. 统计候选词在文档中的出现情况
    3. 利用余弦相似度过滤无关候选词
    4. 对过滤后的候选关键词进行KMeans聚类，
       并在每个簇内独立进行TextRank计算，得到每个关键词的得分
    5. 最终返回包含关键词、所属簇和TextRank得分的聚类结果（只保留得分最高的前N个关键词）
    """
    kw_model = KeyBERT(model=sentence_model)
    keyword_stats = defaultdict(lambda: {'count': 0, 'docs': set()})
    doc_embeddings = []
    
    for idx, text in enumerate(tqdm(texts, desc="处理文档")):
        keywords, doc_emb = process_single_document(text, stopwords, kw_model)
        if not keywords or doc_emb is None:
            continue
        doc_embeddings.append(doc_emb)
        for kw in keywords:
            keyword_stats[kw]['count'] += 1
            keyword_stats[kw]['docs'].add(idx)
    
    if not doc_embeddings:
        print("没有有效的文档嵌入，可能预处理失败。")
        return []
    
    # 对候选关键词进行交叉验证过滤
    all_candidates = list(keyword_stats.keys())
    filtered_keywords = cross_validate_filter(all_candidates, doc_embeddings)
    
    # 保留覆盖率达到最低要求的关键词
    filtered_keywords = [
        kw for kw in filtered_keywords 
        if (len(keyword_stats[kw]['docs']) / len(doc_embeddings)) >= CONFIG['min_doc_coverage']
    ]
    
    if not filtered_keywords:
        print("没有候选关键词满足最低覆盖率要求。")
        return []
    
    candidate_embeddings = sentence_model.encode(filtered_keywords)
    
    # 对候选关键词进行聚类并在每个簇内计算TextRank得分
    cluster_results, cluster_labels = cluster_keywords_textrank(filtered_keywords, candidate_embeddings)
    
    # 将聚类结果拉平，得到列表：(关键词, 簇编号, TextRank得分)
    flat_results = []
    for cluster_id, kw_list in cluster_results.items():
        for kw, score in kw_list:
            flat_results.append((kw, cluster_id, score))
    
    # 全局按TextRank得分降序排序
    flat_results.sort(key=lambda x: x[2], reverse=True)
    
    # 保留得分最高的前N个关键词
    final_results = flat_results[:CONFIG['output_keywords_num']]
    return final_results

def main():
    # 读取Excel文件数据
    df = pd.read_excel(CONFIG['data_path'], sheet_name=CONFIG['sheet_name'])
    raw_texts = df[CONFIG['column_name']].astype(str).tolist()
    raw_texts = [text.strip() for text in raw_texts if text.strip()]
    if not raw_texts:
        print("没有有效数据，请检查Excel文件和指定的列。")
        return
    print(f"成功加载 {len(raw_texts)} 条文本数据。")
    
    # 加载词典
    stopwords, special_words = load_dictionaries()
    if not stopwords or not special_words:
        print("词典加载失败。")
        return
    print("词典加载成功。")
    
    # 提取全局关键词（采用聚类后每个簇内的TextRank打分）
    global_keywords = extract_global_keywords(raw_texts, stopwords)
    print(f"最终提取的关键词数量: {len(global_keywords)}")
    print("前10个关键词及所属簇和TextRank得分:")
    for item in global_keywords[:10]:
        print(item)
    
    # 将关键词及聚类信息保存至Excel文件
    result_df = pd.DataFrame(global_keywords, columns=["关键词", "簇编号", "TextRank得分"])
    result_df.to_excel("改进版全局关键词_聚类_TextRank.xlsx", index=False)
    print("关键词提炼完成，结果已保存！")

if __name__ == "__main__":
    main()
