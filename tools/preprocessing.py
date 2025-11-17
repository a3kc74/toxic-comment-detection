import re
import string
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

# Nếu chưa cài nltk:
# pip install nltk

import nltk
from nltk.corpus import stopwords

# Tải stopwords lần đầu
nltk.download('stopwords')

# -----------------------------
# 1. Chuẩn bị stopwords & slang
# -----------------------------

EN_STOPWORDS = set(stopwords.words('english'))

# Từ điển tiếng lóng / viết tắt phổ biến trong comment tiếng Anh
SLANG_MAP = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "ya": "you",
    "thx": "thanks",
    "pls": "please",
    "plz": "please",
    "btw": "by the way",
    "idk": "i do not know",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "lol": "laughing",
    "lmao": "laughing",
    "wtf": "what the fuck",
    "wth": "what the hell",
    "omg": "oh my god",
    "bc": "because",
    "cuz": "because",
    "kinda": "kind of",
    "wanna": "want to",
    "gonna": "going to",
    "gotta": "got to",
    "dont": "do not",
    "cant": "cannot",
    "wont": "will not",
    # bạn có thể bổ sung thêm dần dần...
}

# -----------------------------
# 2. Các hàm xử lý text
# -----------------------------

def normalize_text(text: str) -> str:
    """
    - lowercase
    - bỏ số
    - bỏ ký tự đặc biệt / dấu câu, chỉ giữ lại chữ + khoảng trắng
    - bỏ khoảng trắng dư
    """
    text = text.lower()
    # bỏ số
    text = re.sub(r'\d+', ' ', text)
    # bỏ ký tự không phải chữ cái hoặc khoảng trắng
    text = re.sub(r'[^a-z\s]', ' ', text)
    # gộp nhiều space thành 1
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str):
    """Tokenize rất đơn giản bằng split (đủ tốt cho bước này)."""
    return text.split()


def normalize_slang(tokens):
    """
    Ánh xạ các từ lóng / viết tắt sang dạng chuẩn.
    Nếu không có trong SLANG_MAP thì giữ nguyên.
    """
    new_tokens = []
    for t in tokens:
        if t in SLANG_MAP:
            # có slang map, có thể là cụm nhiều từ -> tách tiếp
            mapped = SLANG_MAP[t]
            new_tokens.extend(mapped.split())
        else:
            new_tokens.append(t)
    return new_tokens


def remove_stopwords(tokens):
    """Loại bỏ stop words tiếng Anh."""
    return [t for t in tokens if t not in EN_STOPWORDS]


def preprocess_comment(text: str) -> str:
    """
    Pipeline đầy đủ:
      1. normalize text
      2. tokenize
      3. slang normalization
      4. remove stopwords
      5. ghép lại thành chuỗi
    """
    if not isinstance(text, str):
        return ""

    text = normalize_text(text)
    tokens = tokenize(text)
    tokens = normalize_slang(tokens)
    tokens = remove_stopwords(tokens)

    # nếu muốn lưu dạng list token thì return tokens
    # ở đây mình join lại thành string
    return " ".join(tokens)



