import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

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
