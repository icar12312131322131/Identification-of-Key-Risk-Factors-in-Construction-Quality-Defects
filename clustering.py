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
# Configuration
class Config:
    DATA_PATH = r"filepath"  # Dataset path
    SHEET_NAME = " "                                     # Sheet name
    TEXT_COLUMN = " "                                   # Column with text
    STOPWORDS_PATH = r" "          # Stopwords
    DOMAIN_WORDS_PATH = r" "           # Domain-specific words
    MIN_TOPICS = 5
    MAX_TOPICS = 5
    PASSES = 500
    RANDOM_SEED = 42
    AUTOENCODER_EPOCHS = 100
    BATCH_SIZE = 64
    ENCODING_DIM = 64
    CLUSTER_PLOT_SAMPLE = 3800
    NUM_WORKERS = max(1, multiprocessing.cpu_count() // 2)
    EMBEDDING_MODEL = "moka-ai/m3e-large"
    CANOPY_SAMPLE_SIZE = 5000

cfg = Config()

# -------------------
# Utilities
def load_stopwords(path: str) -> Set[str]:
    """Load stopwords, supporting multiple encodings"""
    encodings = ['utf-8', 'gbk', 'latin1']
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                return {line.strip() for line in f if line.strip()}
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode file: {path}")

def load_domain_words(path: str) -> Set[str]:
    """Load domain words and add to Jieba dictionary"""
    words = load_stopwords(path)
    for word in words:
        jieba.add_word(word, freq=10000)
    return words

def preprocess(text: str, stopwords: Set[str], domain_words: Set[str]) -> List[str]:
    """Text preprocessing: clean, tokenize, POS filtering"""
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip().lower()
    
    words = []
    for word, flag in pseg.cut(text):
        if (word in domain_words) or (len(word) > 1 and word not in stopwords and flag not in ['x', 'w', 'uj']):
            words.append(word)
    return words

def compute_metrics(args: Tuple[List[Any], corpora.Dictionary, List[List[str]], int]) -> Tuple[int, float, float, Any]:
    """Compute LDA metrics: perplexity and coherence"""
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
        print(f"Topic {num_topics} error: {str(e)}")
        return num_topics, None, None, None

def build_sbert_model() -> SentenceTransformer:
    """Build SBERT embedding model"""
    try:
        word_embedding = st_models.Transformer(cfg.EMBEDDING_MODEL, model_args={"trust_remote_code": True})
        pooling = st_models.Pooling(word_embedding.get_word_embedding_dimension(), pooling_mode='mean')
        return SentenceTransformer(modules=[word_embedding, pooling])
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def parallel_lda_processing(texts: List[List[str]]) -> Tuple[corpora.Dictionary, List[List[Tuple[int, int]]]]:
    """Construct dictionary and corpus for LDA"""
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=max(5, int(len(texts)*0.001)), no_above=0.5, keep_n=100000)
    corpus = [dictionary.doc2bow(doc) for doc in texts]
    return dictionary, corpus

def enhanced_autoencoder(input_dim: int) -> Tuple[Model, Model]:
    """Build autoencoder"""
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
    """Improved density-based Canopy clustering"""
    if len(X) > cfg.CANOPY_SAMPLE_SIZE:
        sample_indices = np.random.choice(len(X), cfg.CANOPY_SAMPLE_SIZE, replace=False)
        X_sample = X[sample_indices]
    else:
        X_sample = X

    nbrs = NearestNeighbors(n_neighbors=3800, algorithm='kd_tree').fit(X_sample)
    distances, indices_knn = nbrs.kneighbors(X_sample)
    mean_dis = np.mean(distances[:, 1:]) * 1.32

    rho = np.sum(distances < mean_dis, axis=1)
    a = np.zeros(len(X_sample))
    for i in range(len(X_sample)):
        neighbors = np.where((distances[i] < mean_dis) & (indices_knn[i] != i))[0]
        if len(neighbors) > 1:
            sub_dist = distances[np.ix_(neighbors, neighbors)]
            a[i] = np.sum(sub_dist) / (len(neighbors) * (len(neighbors) - 1))
    
    full_dists = np.linalg.norm(X_sample[:, None] - X_sample[None, :], axis=2)
    delta = np.zeros(len(X_sample))
    for i in range(len(X_sample)):
        higher_rho = np.where(rho > rho[i])[0]
        delta[i] = np.min(full_dists[i, higher_rho]) if len(higher_rho) > 0 else np.max(full_dists[i, :])
    
    epsilon = 1e-6
    w = rho * (1.0 / (a + epsilon)) * delta

    remaining = np.arange(len(X_sample))
    centers = []
    while len(remaining) > 0:
        idx = remaining[np.argmax(w[remaining])]
        centers.append(X_sample[idx])
        center_dists = np.linalg.norm(X_sample - X_sample[idx], axis=1)
        mask = center_dists[remaining] <= mean_dis
        remaining = remaining[~mask]

    k = len(centers) if centers else 1
    if not centers:
        centers = [X_sample[np.argmax(rho)]]
    
    kmeans = KMeans(n_clusters=k, init=np.array(centers), n_init=1, random_state=random_state)
    full_labels = kmeans.fit_predict(X)
    return full_labels, k

def lda_vector_generator(corpus: List[List[Tuple[int, int]]], lda_model: Any) -> np.ndarray:
    """Generate LDA topic vectors with uniform dimension"""
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
    """Visualize metrics, autoencoder loss, and clustering"""
    plt.figure(figsize=(20, 7), dpi=190)
    
    # LDA metrics
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(topics, perplexities, 'b-o', markersize=8)
    ax1.set_xlabel("Number of Topics", fontsize=12)
    ax2 = ax1.twinx()
    ax2.plot(topics, coherences, 'r-*', markersize=10)
    ax2.set_ylabel("Coherence", color='r', fontsize=12)
    plt.title("LDA Model Metrics", fontsize=14)
    
    # Autoencoder loss
    plt.subplot(1, 3, 2)
    plt.semilogy(history.history['loss'], label='Train')
    plt.semilogy(history.history['val_loss'], label='Validation')
    plt.title("Autoencoder Training Curve", fontsize=14)
    
    # Clustering visualization
    plt.subplot(1, 3, 3)
    sample_size = min(cfg.CLUSTER_PLOT_SAMPLE, len(latent))
    sample_idx = np.random.choice(len(latent), sample_size, replace=False)
    tsne = TSNE(n_components=2, perplexity=30, random_state=cfg.RANDOM_SEED)
    vis_data = tsne.fit_transform(latent[sample_idx])
    sc = plt.scatter(vis_data[:, 0], vis_data[:, 1], c=labels[sample_idx], cmap='Spectral', alpha=0.7)
    cbar = plt.colorbar(sc, boundaries=np.arange(np.max(labels)+2)-0.5)
    cbar.set_ticks(np.arange(np.max(labels)+1))
    plt.title("Clustering Visualization", fontsize=14)
    
    plt.tight_layout()
    plt.savefig("analysis_results.jpg", bbox_inches='tight')
    plt.close()

def save_results(raw_texts: List[str], processed_texts: List[List[str]], labels: np.ndarray,
                 lda_model: Any, corpus: List[List[Tuple[int, int]]]) -> None:
    """Save detailed and summary results"""
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
    
    result_df = pd.DataFrame({
        "Raw Text": raw_texts[:len(processed_texts)],
        "Tokenized Text": [" ".join(t) for t in processed_texts],
        "Cluster Label": labels,
        "Topic Keywords": [topic_keywords[label] for label in labels]
    })
    
    summary_df = pd.DataFrame({
        "Cluster Label": range(len(topic_keywords)),
        "Sample Count": result_df.groupby("Cluster Label").size(),
        "Representative Keywords": topic_keywords
    })
    
    with pd.ExcelWriter("clustering_results.xlsx") as writer:
        result_df.to_excel(writer, sheet_name="Detailed Results", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
    
    print("💾 Results saved to clustering_results.xlsx")

# -------------------
# Main workflow
def main() -> None:
    # 1. Load data
    print("🔄 [1/8] Loading dataset...")
    try:
        df = pd.read_excel(cfg.DATA_PATH, sheet_name=cfg.SHEET_NAME,
                           usecols=[cfg.TEXT_COLUMN], engine='openpyxl')
        texts_raw = df.iloc[:, 0].astype(str).str.strip().tolist()
        print(f"📊 Initial number of texts: {len(texts_raw)}")
    except Exception as e:
        print(f"❌ Data loading error: {repr(e)}")
        return

    # 2. Load language resources
    print("🔄 [2/8] Loading language resources...")
    try:
        stopwords = load_stopwords(cfg.STOPWORDS_PATH)
        domain_words = load_domain_words(cfg.DOMAIN_WORDS_PATH)
    except Exception as e:
        print(f"❌ Resource loading error: {str(e)}")
        return

    # 3. Preprocessing
    print("🔄 [3/8] Preprocessing texts in parallel...")
    valid_indices = []
    processed_texts = []
    with multiprocessing.Pool(cfg.NUM_WORKERS) as pool:
        tasks = [(t, stopwords, domain_words) for t in texts_raw]
        for idx, result in enumerate(tqdm(pool.starmap(preprocess, tasks), desc="Preprocessing")):
            if result and len(result) >= 1:
                processed_texts.append(result)
                valid_indices.append(idx)
    
    texts_raw = [texts_raw[i] for i in valid_indices]
    print(f"✅ Valid texts: {len(processed_texts)}")

    # 4. LDA modeling
    print("🔄 [4/8] Training LDA models...")
    dictionary, corpus = parallel_lda_processing(processed_texts)
    
    with multiprocessing.Pool(cfg.NUM_WORKERS) as pool:
        tasks = [(corpus, dictionary, processed_texts, n)
                 for n in range(cfg.MIN_TOPICS, cfg.MAX_TOPICS + 1)]
        results = []
        for res in tqdm(pool.imap(compute_metrics, tasks), total=len(tasks), desc="LDA computation"):
            if res[2] is not None:
                results.append(res)
    
    if not results:
        print("❌ No valid LDA models found")
        return
    topics, perplexities, coherences, lda_models = zip(*sorted(results, key=lambda x: x[0]))
    best_idx = np.nanargmax(coherences)
    best_lda = lda_models[best_idx]
    best_topic = topics[best_idx]
    print(f"🎯 Optimal number of topics: {best_topic} (Coherence: {coherences[best_idx]:.3f})")

    # 5. SBERT embedding
    print("🔄 [5/8] Generating embeddings...")
    try:
        sbert_model = build_sbert_model()
        sentences = [" ".join(t) for t in processed_texts]
        sbert_embeddings = sbert_model.encode(
            sentences, batch_size=cfg.BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True
        )
    except Exception as e:
        print(f"❌ Embedding generation failed: {str(e)}")
        return

    # 6. Feature fusion
    print("🔄 [6/8] Feature fusion...")
    lda_embeddings = lda_vector_generator(corpus, best_lda)
    combined = np.hstack((sbert_embeddings, lda_embeddings))
    scaler = MinMaxScaler()
    combined_scaled = scaler.fit_transform(combined)

    # 7. Autoencoder
    print("🔄 [7/8] Training autoencoder...")
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

    # 8. Clustering
    print("🔄 [8/8] Performing clustering analysis...")
    labels, k = density_canopy_kmeans(latent, random_state=cfg.RANDOM_SEED)
    
    # Visualization
    visualize_results(history, latent, labels, list(topics), list(perplexities), list(coherences), best_topic)

    # Save results
    save_results(texts_raw, processed_texts, labels, best_lda, corpus)

if __name__ == "__main__":
    main()

