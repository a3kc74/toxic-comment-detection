import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    confusion_matrix, classification_report
)


def get_top_ngrams(texts, ngram_range=(1, 1), top_k=20, min_df=5):
    """
    texts: iterable các comment (chuỗi)
    ngram_range: (1,1) -> unigram; (2,2) -> bigram
    top_k: số n-gram muốn lấy
    min_df: bỏ những n-gram xuất hiện quá ít
    """
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        min_df=min_df
    )
    X = vectorizer.fit_transform(texts)
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    
    sorted_idx = np.argsort(sums)[::-1]
    top_terms = terms[sorted_idx][:top_k]
    top_counts = sums[sorted_idx][:top_k]
    
    return list(zip(top_terms, top_counts))

# -----------------------------
# 1) Tạo & fit TF-IDF
# -----------------------------
def build_tfidf_vectorizer(
    X_train,
    X_valid=None,
    max_features=100_000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    norm='l2'
):
    """
    Fit TfidfVectorizer trên X_train, 
    nếu có X_valid thì transform luôn.
    Trả về: tfidf, X_train_tfidf, X_valid_tfidf (hoặc None)
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        norm=norm
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    
    if X_valid is not None:
        X_valid_tfidf = tfidf.transform(X_valid)
    else:
        X_valid_tfidf = None
    
    return tfidf, X_train_tfidf, X_valid_tfidf


# -----------------------------
# 2) Train model LinearRegression với pipeline
# -----------------------------
def train_linear_model(X_train, y_train, tfidf=None):
    """
    Train LinearRegression multi-output.
    Nếu tfidf != None: dùng tfidf đã fit.
    Nếu tfidf == None: pipeline sẽ tự tạo TfidfVectorizer mới.
    Trả về: model (pipeline)
    """
    if tfidf is None:
        # trường hợp muốn pipeline tự fit tfidf từ X_train dạng text
        tfidf = TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True,
            norm='l2'
        )

    model = make_pipeline(
        tfidf,
        LinearRegression()
    )

    model.fit(X_train, y_train)
    return model


# -----------------------------
# 3) Dự đoán với model đã train
# -----------------------------
def predict_with_model(model, X_valid):
    """
    Dự đoán y_pred từ model (pipeline).
    Trả về: y_pred, y_pred_binary, y_pred_probs
    """
    y_pred = model.predict(X_valid)       # (n_valid, 7)
    y_pred_binary = y_pred[:, 0]         # non_toxic
    y_pred_probs = y_pred[:, 1:]         # 6 nhãn còn lại
    return y_pred, y_pred_binary, y_pred_probs


# -----------------------------
# 4) Evaluate & vẽ heatmap cho nhãn binary
# -----------------------------
def evaluate_binary_label(y_valid, y_pred, label_name="non_toxic"):
    """
    - Tính MAE/MSE trên full y (multi-output)
    - Confusion matrix + heatmap cho nhãn binary (cột 0)
    - Classification report cho nhãn binary
    """
    # y_valid, y_pred: (n_samples, 7)
    # tách nhãn binary
    y_valid_binary = y_valid[:, 0]
    y_pred_binary = y_pred[:, 0]

    # ----- Regression metric cho toàn bộ 7 cột -----
    mae = mean_absolute_error(y_valid, y_pred)
    mse = mean_squared_error(y_valid, y_pred)

    print(f"[Multi-output] MAE: {mae:.4f}")
    print(f"[Multi-output] MSE: {mse:.4f}\n")

    # ----- Confusion matrix cho binary -----
    y_pred_binary_class = (y_pred_binary >= 0.5).astype(int)
    cm = confusion_matrix(y_valid_binary, y_pred_binary_class)
    print(f"Confusion matrix for {label_name}:")
    print(cm)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix – {label_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"confusion_matrix_{label_name}.png", dpi=300)
    plt.show()

    # ----- Classification report -----
    print(f"\nClassification report for {label_name}:")
    print(classification_report(y_valid_binary, y_pred_binary_class))
