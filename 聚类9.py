import pandas as pd
import jieba
import jieba.posseg as pseg
from gensim import corpora, models
import matplotlib.pyplot as plt
import re
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models as st_models
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import openpyxl
import multiprocessing
from warnings import filterwarnings
from typing import List, Set, Tuple, Any, Dict
from collections import defaultdict

filterwarnings('ignore')

# -------------------
# 增强配置参数
class Config:
    DATA_PATH = r"D:\东西中东北\东部_combined_by_period.xlsx"          # 数据路径
    SHEET_NAME = "Sheet1"                  # 工作表名称
    TEXT_COLUMN = "sentence"                     # 文本列（A列）
    STOPWORDS_PATH = r"D:\文本挖掘\hit_stopwords.txt"   # HIT停用词
    DOMAIN_WORDS_PATH = r"D:\文本挖掘\人文政治.txt"  # 领域专用词    
    MIN_TOPICS = 5                       # 最小主题数
    MAX_TOPICS = 5                       # 最大主题数
    PASSES = 500                          # LDA训练轮次
    RANDOM_SEED = 42                      # 随机种子
    AUTOENCODER_EPOCHS = 100              # 自动编码器训练轮次
    BATCH_SIZE = 64                       # 批量大小
    ENCODING_DIM = 64                     # 潜在空间维度
    CLUSTER_PLOT_SAMPLE = 3800            # 聚类可视化采样数
    NUM_WORKERS = max(1, multiprocessing.cpu_count() // 2)  # 并行进程数
    EMBEDDING_MODEL = "moka-ai/m3e-large"  # 嵌入模型
    CANOPY_SAMPLE_SIZE = 5000             # 密度Canopy采样大小

cfg = Config()

# -------------------
# 工具函数
def load_stopwords(path: str) -> Set[str]:
    """加载停用词，支持多种编码格式"""
    encodings = ['utf-8', 'gbk', 'latin1']
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                return {line.strip() for line in f if line.strip()}
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解码文件: {path}")

def load_domain_words(path: str) -> Set[str]:
    """加载领域词典，并添加到jieba中以提高切分准确率"""
    words = load_stopwords(path)
    for word in words:
        jieba.add_word(word, freq=10000)
    return words

def preprocess(text: str, stopwords: Set[str], domain_words: Set[str]) -> List[str]:
    """文本预处理：清洗、分词、词性标注"""
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip().lower()
    
    words = []
    for word, flag in pseg.cut(text):
        keep_cond = (
            (word in domain_words) or
            (len(word) > 1 and word not in stopwords and flag not in ['x', 'w', 'uj'])
        )
        if keep_cond:
            words.append(word)
    return words

def compute_metrics(args: Tuple[List[Any], corpora.Dictionary, List[List[str]], int]) -> Tuple[int, float, float, Any]:
    """计算LDA模型指标"""
    corpus, dictionary, texts, num_topics = args
    try:
        lda = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=cfg.PASSES,
            random_state=cfg.RANDOM_SEED,
            alpha='auto'
        )
        perplexity = -lda.log_perplexity(corpus)
        coherence = models.CoherenceModel(
            model=lda, texts=texts, dictionary=dictionary, 
            coherence='c_v', processes=1
        ).get_coherence()
        return num_topics, perplexity, coherence, lda
    except Exception as e:
        print(f"主题数 {num_topics} 出错: {str(e)}")
        return num_topics, None, None, None

def build_sbert_model() -> SentenceTransformer:
    """构建SBERT嵌入模型"""
    try:
        word_embedding = st_models.Transformer(
            cfg.EMBEDDING_MODEL, 
            model_args={"trust_remote_code": True}
        )
        pooling = st_models.Pooling(
            word_embedding.get_word_embedding_dimension(),
            pooling_mode='mean'
        )
        return SentenceTransformer(modules=[word_embedding, pooling])
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}")

def parallel_lda_processing(texts: List[List[str]]) -> Tuple[corpora.Dictionary, List[List[Tuple[int, int]]]]:
    """构建词典和语料"""
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(
        no_below=max(5, int(len(texts) * 0.001)),
        no_above=0.5,
        keep_n=100000
    )
    corpus = [dictionary.doc2bow(doc) for doc in texts]
    return dictionary, corpus

def enhanced_autoencoder(input_dim: int) -> Tuple[Model, Model]:
    """构建自动编码器"""
    input_layer = Input(shape=(input_dim,))
    x = Dense(512, activation='swish')(input_layer)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='swish')(x)
    x = Dropout(0.2)(x)
    encoded = Dense(cfg.ENCODING_DIM, activation='swish')(x)
    
    x = Dense(256, activation='swish')(encoded)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='swish')(x)
    x = Dropout(0.3)(x)
    decoded = Dense(input_dim, activation='linear')(x)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, Model(input_layer, encoded)

def density_canopy_kmeans(X: np.ndarray, random_state: int = None) -> Tuple[np.ndarray, int]:
    """改进的密度Canopy算法"""
    if len(X) > cfg.CANOPY_SAMPLE_SIZE:
        sample_indices = np.random.choice(len(X), cfg.CANOPY_SAMPLE_SIZE, replace=False)
        X_sample = X[sample_indices]
    else:
        X_sample = X

    nbrs = NearestNeighbors(n_neighbors=3800, algorithm='kd_tree').fit(X_sample)
    # 同时返回距离和对应的索引
    distances, indices_knn = nbrs.kneighbors(X_sample)
    mean_dis = np.mean(distances[:, 1:])*1.32  # 忽略自身（第一个距离为0）

    rho = np.sum(distances < mean_dis, axis=1)
    a = np.zeros(len(X_sample))
    for i in range(len(X_sample)):
        # 使用返回的索引数组，保证比较的数组形状一致
        neighbors = np.where((distances[i] < mean_dis) & (indices_knn[i] != i))[0]
        if len(neighbors) > 1:
            sub_dist = distances[np.ix_(neighbors, neighbors)]
            a[i] = np.sum(sub_dist) / (len(neighbors) * (len(neighbors) - 1))
    
    # 使用全距离矩阵来计算delta，避免超出预计算邻居距离的范围
    full_dists = np.linalg.norm(X_sample[:, None] - X_sample[None, :], axis=2)
    delta = np.zeros(len(X_sample))
    for i in range(len(X_sample)):
        higher_rho = np.where(rho > rho[i])[0]
        if len(higher_rho) > 0:
            delta[i] = np.min(full_dists[i, higher_rho])
        else:
            delta[i] = np.max(full_dists[i, :])
    
    epsilon = 1e-6
    w = rho * (1.0 / (a + epsilon)) * delta

    remaining = np.arange(len(X_sample))
    centers = []
    while len(remaining) > 0:
        idx = remaining[np.argmax(w[remaining])]
        centers.append(X_sample[idx])
        # 计算选中中心与所有样本之间的欧式距离
        center_dists = np.linalg.norm(X_sample - X_sample[idx], axis=1)
        mask = center_dists[remaining] <= mean_dis
        remaining = remaining[~mask]

    k = len(centers) if centers else 1
    if not centers:
        centers = [X_sample[np.argmax(rho)]]
    
    # 使用完整数据进行K-means聚类
    kmeans = KMeans(n_clusters=k, init=np.array(centers), n_init=1, random_state=random_state)
    full_labels = kmeans.fit_predict(X)
    return full_labels, k

def lda_vector_generator(corpus: List[List[Tuple[int, int]]], lda_model: Any) -> np.ndarray:
    """生成统一维度的LDA向量"""
    num_topics = lda_model.num_topics
    vectors = []
    for doc in corpus:
        topic_probs = dict(lda_model.get_document_topics(doc, minimum_probability=0))
        vec = [topic_probs.get(i, 0.0) for i in range(num_topics)]
        vectors.append(vec)
    return np.array(vectors)

def visualize_results(history: Any, latent: np.ndarray, labels: np.ndarray,
                     topics: List[int], perplexities: List[float], coherences: List[float],
                     best_topic: int) -> None:
    """可视化结果"""
    plt.figure(figsize=(20, 7), dpi=190)
    
    # LDA指标
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(topics, perplexities, 'b-o', markersize=8)
    ax1.set_xlabel("主题数量", fontsize=12)
    ax2 = ax1.twinx()
    ax2.plot(topics, coherences, 'r-*', markersize=10)
    ax2.set_ylabel("一致性", color='r', fontsize=12)
    plt.title("LDA模型指标趋势", fontsize=14)
    
    # 自编码器损失
    plt.subplot(1, 3, 2)
    plt.semilogy(history.history['loss'], label='训练集')
    plt.semilogy(history.history['val_loss'], label='验证集')
    plt.title("自动编码器学习曲线", fontsize=14)
    
    # 聚类可视化
    plt.subplot(1, 3, 3)
    sample_size = min(cfg.CLUSTER_PLOT_SAMPLE, len(latent))
    sample_idx = np.random.choice(len(latent), sample_size, replace=False)
    tsne = TSNE(n_components=2, perplexity=30, random_state=cfg.RANDOM_SEED)
    vis_data = tsne.fit_transform(latent[sample_idx])
    
    sc = plt.scatter(vis_data[:, 0], vis_data[:, 1],
                     c=labels[sample_idx], cmap='Spectral', alpha=0.7)
    cbar = plt.colorbar(sc, boundaries=np.arange(np.max(labels)+2)-0.5)
    cbar.set_ticks(np.arange(np.max(labels)+1))
    plt.title("聚类分布可视化", fontsize=14)
    
    plt.tight_layout()
    plt.savefig("analysis_results.jpg", bbox_inches='tight')
    plt.close()

def save_results(raw_texts: List[str], processed_texts: List[List[str]], labels: np.ndarray,
                 lda_model: Any, corpus: List[List[Tuple[int, int]]]) -> None:
    """保存结果"""
    # 生成聚类关键词
    cluster_words = defaultdict(list)
    for label, words in zip(labels, processed_texts):
        cluster_words[label].extend(words)
    
    topic_keywords = []
    for cluster in sorted(cluster_words.keys()):
        word_freq = defaultdict(int)
        for word in cluster_words[cluster]:
            word_freq[word] += 1
        top_words = [w for w, _ in sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))[:90]]
        topic_keywords.append(", ".join(top_words))
    
    # 生成DataFrame
    result_df = pd.DataFrame({
        "原始文本": raw_texts[:len(processed_texts)],
        "分词结果": [" ".join(t) for t in processed_texts],
        "聚类标签": labels,
        "主题关键词": [topic_keywords[label] for label in labels]
    })
    
    summary_df = pd.DataFrame({
        "聚类标签": range(len(topic_keywords)),
        "样本数量": result_df.groupby("聚类标签").size(),
        "代表性关键词": topic_keywords
    })
    
    with pd.ExcelWriter("聚类分析结果.xlsx") as writer:
        result_df.to_excel(writer, sheet_name="详细结果", index=False)
        summary_df.to_excel(writer, sheet_name="聚类摘要", index=False)
    
    print("💾 结果已保存到聚类分析结果.xlsx")

# -------------------
# 主流程
def main() -> None:
    # 1. 数据加载
    print("🔄 [1/8] 加载数据...")
    try:
        df = pd.read_excel(cfg.DATA_PATH, sheet_name=cfg.SHEET_NAME,
                           usecols=[cfg.TEXT_COLUMN], engine='openpyxl')
        texts_raw = df.iloc[:, 0].astype(str).str.strip().tolist()
        print(f"📊 初始文本数: {len(texts_raw)}")
    except Exception as e:
        print(f"❌ 数据加载错误: {repr(e)}")
        return

    # 2. 加载语言资源
    print("🔄 [2/8] 加载语言资源...")
    try:
        stopwords = load_stopwords(cfg.STOPWORDS_PATH)
        domain_words = load_domain_words(cfg.DOMAIN_WORDS_PATH)
    except Exception as e:
        print(f"❌ 资源加载错误: {str(e)}")
        return

    # 3. 预处理
    print("🔄 [3/8] 并行预处理文本...")
    valid_indices = []  # 记录有效文本的索引
    processed_texts = []
    with multiprocessing.Pool(cfg.NUM_WORKERS) as pool:
        tasks = [(t, stopwords, domain_words) for t in texts_raw]
        for idx, result in enumerate(tqdm(pool.starmap(preprocess, tasks), desc="预处理进度")):
            if result and len(result) >= 1:  # 至少保留一个有效词
                processed_texts.append(result)
                valid_indices.append(idx)
    
    # 仅保留有效原始文本
    texts_raw = [texts_raw[i] for i in valid_indices]
    print(f"✅ 有效文本: {len(processed_texts)}")

    # 4. LDA训练
    print("🔄 [4/8] 训练LDA模型...")
    dictionary, corpus = parallel_lda_processing(processed_texts)
    
    with multiprocessing.Pool(cfg.NUM_WORKERS) as pool:
        tasks = [(corpus, dictionary, processed_texts, n)
                 for n in range(cfg.MIN_TOPICS, cfg.MAX_TOPICS + 1)]
        results = []
        for res in tqdm(pool.imap(compute_metrics, tasks), total=len(tasks), desc="LDA计算"):
            if res[2] is not None:
                results.append(res)
    
    if not results:
        print("❌ 无有效LDA模型")
        return
    topics, perplexities, coherences, lda_models = zip(*sorted(results, key=lambda x: x[0]))
    best_idx = np.nanargmax(coherences)
    best_lda = lda_models[best_idx]
    best_topic = topics[best_idx]
    print(f"🎯 最优主题数: {best_topic} (一致性: {coherences[best_idx]:.3f})")

    # 5. SBERT嵌入
    print("🔄 [5/8] 生成嵌入向量...")
    try:
        sbert_model = build_sbert_model()
        sentences = [" ".join(t) for t in processed_texts]
        sbert_embeddings = sbert_model.encode(
            sentences,
            batch_size=cfg.BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )
    except Exception as e:
        print(f"❌ 嵌入生成失败: {str(e)}")
        return

    # 6. 特征融合
    print("🔄 [6/8] 特征融合...")
    lda_embeddings = lda_vector_generator(corpus, best_lda)
    assert sbert_embeddings.shape[0] == lda_embeddings.shape[0], \
        f"样本数不匹配: SBERT={sbert_embeddings.shape[0]}, LDA={lda_embeddings.shape[0]}"
    combined = np.hstack((sbert_embeddings, lda_embeddings))
    print(f"SBERT嵌入形状: {sbert_embeddings.shape}")
    print(f"LDA向量形状: {lda_embeddings.shape}")
    print(f"合并后形状: {combined.shape}")
    scaler = MinMaxScaler()
    combined_scaled = scaler.fit_transform(combined)

    # 7. 自编码器
    print("🔄 [7/8] 训练自动编码器...")
    autoencoder, encoder = enhanced_autoencoder(combined_scaled.shape[1])
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    history = autoencoder.fit(
        combined_scaled, combined_scaled,
        epochs=cfg.AUTOENCODER_EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )
    latent = encoder.predict(combined_scaled, verbose=0)

    # 8. 聚类分析
    print("🔄 [8/8] 进行聚类分析...")
    labels, k = density_canopy_kmeans(latent, random_state=cfg.RANDOM_SEED)
   
    # 可视化结果
    visualize_results(history, latent, labels, list(topics), list(perplexities), list(coherences), best_topic)

    # 保存最终结果
    save_results(texts_raw, processed_texts, labels, best_lda, corpus)

if __name__ == "__main__":
    main()
