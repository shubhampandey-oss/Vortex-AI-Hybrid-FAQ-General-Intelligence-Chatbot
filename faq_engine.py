"""
faq_engine.py
=============================================================================
FAQ Engine: TF-IDF Cosine Similarity & Dynamic Knowledge Base Manager
-----------------------------------------------------------------------------
Features:
- Dual-channel vectorizer (Word n-grams (1,2) + Char-wb n-grams (3,5) for typo resilience)
- Cosine similarity calculation with configurable threshold
- Direct alias and synonym matcher
- Live dynamic CRUD operations (Add, Edit, Delete, Search) with real-time vectorizer refitting
=============================================================================
"""

import json
import logging
import re
import string
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config

logger = logging.getLogger("FAQEngine")

# Safe NLP Tokenization and Stemming/Lemmatizing without network blocking
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    RAW_STOP_WORDS = set(stopwords.words("english"))
    PRESERVED_TOKENS = {
        "what", "how", "why", "who", "which", "where", "when",
        "difference", "between", "vs", "versus", "define", "explain",
        "write", "code", "python", "program", "not", "no", "more", "most"
    }
    STOP_WORDS = RAW_STOP_WORDS - PRESERVED_TOKENS
    NLTK_TOKENIZE_AVAILABLE = True
except Exception:
    NLTK_TOKENIZE_AVAILABLE = False
    STOP_WORDS = {
        "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she", "it",
        "they", "them", "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "a", "an", "the", "and", "but",
        "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
        "about", "into", "through", "during", "before", "after", "to", "from", "in", "out"
    }

# Safe Stemmer / Lemmatizer (PorterStemmer never requires extra network downloads)
STEMMER = None
try:
    from nltk.stem import WordNetLemmatizer
    _lem = WordNetLemmatizer()
    _lem.lemmatize("testing")  # Test if wordnet resource is locally present
    STEMMER = _lem
    LEMMATIZE_METHOD = "lemmatize"
except Exception:
    try:
        from nltk.stem import PorterStemmer
        STEMMER = PorterStemmer()
        LEMMATIZE_METHOD = "stem"
    except Exception:
        STEMMER = None
        LEMMATIZE_METHOD = None


def preprocess_text(text: str) -> str:
    """Preprocesses input text: lowercasing, punctuation stripping, tokenization, lemmatizing/stemming."""
    if not text or not isinstance(text, str):
        return ""

    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("-", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))

    if NLTK_TOKENIZE_AVAILABLE:
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
    else:
        tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", text)

    processed = []
    for word in tokens:
        if word not in STOP_WORDS and len(word) > 1:
            if STEMMER and LEMMATIZE_METHOD == "lemmatize":
                try:
                    word = STEMMER.lemmatize(word)
                except Exception:
                    pass
            elif STEMMER and LEMMATIZE_METHOD == "stem":
                try:
                    word = STEMMER.stem(word)
                except Exception:
                    pass
            processed.append(word)

    return " ".join(processed) if processed else text


class FAQEngine:
    """
    Scikit-learn TF-IDF + Cosine Similarity FAQ Engine with dynamic CRUD and re-fitting.
    """

    def __init__(self, faqs_path: Optional[Path] = None, default_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD):
        self.faqs_path = faqs_path or config.FAQS_FILE
        self.default_threshold = default_threshold

        self.faqs: List[Dict[str, Any]] = []
        self.corpus_texts: List[str] = []

        # Vectorizers and matrices
        self.word_vectorizer: Optional[TfidfVectorizer] = None
        self.char_vectorizer: Optional[TfidfVectorizer] = None
        self.word_matrix = None
        self.char_matrix = None

        self.load_and_fit()

    def load_and_fit(self) -> int:
        """Loads FAQs from disk and fits both Word and Char-wb TF-IDF vectorizers."""
        if not self.faqs_path.exists():
            logger.warning(f"FAQ file not found at {self.faqs_path}. Initializing empty dataset.")
            self.faqs = []
            return 0

        try:
            with open(self.faqs_path, "r", encoding="utf-8") as f:
                self.faqs = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {self.faqs_path}: {e}")
            self.faqs = []
            return 0

        self.corpus_texts = []
        for item in self.faqs:
            q = item.get("question", "")
            aliases = " ".join(item.get("aliases", []))
            tags = " ".join(item.get("tags", []))
            doc = f"{q} {aliases} {tags}"
            self.corpus_texts.append(preprocess_text(doc))

        if not self.corpus_texts:
            self.word_vectorizer = None
            self.char_vectorizer = None
            self.word_matrix = None
            self.char_matrix = None
            return 0

        # Word TF-IDF (unigrams + bigrams)
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2"
        )
        self.word_matrix = self.word_vectorizer.fit_transform(self.corpus_texts)

        # Char-wb TF-IDF (3 to 5 chars for typo tolerance)
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            norm="l2"
        )
        self.char_matrix = self.char_vectorizer.fit_transform(self.corpus_texts)

        logger.info(f"Fitted FAQ Engine with {len(self.faqs)} FAQs. Vocabulary: {len(self.word_vectorizer.vocabulary_)}")
        return len(self.faqs)

    def match_alias_exact(self, query: str) -> Optional[Tuple[Dict[str, Any], float]]:
        """Scans all FAQ alias variations for an exact or near-exact high-confidence match."""
        clean_q = query.lower().strip().rstrip("?!. ")

        # 1. Direct exact alias match
        for item in self.faqs:
            for alias in item.get("aliases", []):
                clean_alias = alias.lower().strip().rstrip("?!. ")
                if clean_q == clean_alias:
                    return item, 0.98

        # 2. Token overlap substring match
        for item in self.faqs:
            for alias in item.get("aliases", []):
                clean_alias = alias.lower().strip().rstrip("?!. ")
                if len(clean_alias) >= 5 and (clean_q == clean_alias or clean_q.startswith(clean_alias) or clean_alias in clean_q):
                    q_words = set(clean_q.split())
                    a_words = set(clean_alias.split())
                    if a_words.issubset(q_words) and len(a_words) >= 2:
                        return item, 0.92

        return None

    def match_query(
        self,
        query: str,
        category_filter: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], float, List[Dict[str, Any]]]:
        """
        Calculates cosine similarity of query against FAQ knowledge base.
        Returns: (best_faq, best_score, top_candidates)
        """
        if not self.faqs or self.word_vectorizer is None or self.char_vectorizer is None:
            return None, 0.0, []

        # 1. Alias exact check
        alias_res = self.match_alias_exact(query)
        if alias_res:
            item, score = alias_res
            if not category_filter or category_filter == "all" or item.get("category") == category_filter:
                candidate = dict(item)
                candidate["similarity_score"] = score
                return item, score, [candidate]

        # 2. Dual-channel TF-IDF Vectorization
        cleaned_query = preprocess_text(query)
        if not cleaned_query:
            cleaned_query = query.lower().strip()

        q_word_vec = self.word_vectorizer.transform([cleaned_query])
        q_char_vec = self.char_vectorizer.transform([query.lower().strip()])

        word_sims = cosine_similarity(q_word_vec, self.word_matrix).flatten()
        char_sims = cosine_similarity(q_char_vec, self.char_matrix).flatten()

        # Weighted combination: 70% word semantics + 30% character n-gram typo tolerance
        combined_sims = (0.70 * word_sims) + (0.30 * char_sims)

        # Typo rescue: If word match is near zero due to misspelling, allow char n-grams to rescue
        max_word = float(np.max(word_sims)) if len(word_sims) > 0 else 0.0
        if max_word < 0.15:
            combined_sims = np.maximum(combined_sims, char_sims)

        # Category filter
        if category_filter and category_filter != "all":
            for idx, item in enumerate(self.faqs):
                if item.get("category") != category_filter:
                    combined_sims[idx] = -1.0

        ranked_indices = np.argsort(combined_sims)[::-1]
        best_idx = ranked_indices[0]
        best_score = float(combined_sims[best_idx])

        candidates = []
        for idx in ranked_indices[:3]:
            score = float(combined_sims[idx])
            if score > 0.05:
                faq_copy = dict(self.faqs[idx])
                faq_copy["similarity_score"] = round(score, 4)
                candidates.append(faq_copy)

        if best_score <= 0.0:
            return None, 0.0, candidates

        return self.faqs[best_idx], best_score, candidates

    def get_all_faqs(self, category: Optional[str] = None, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns all FAQs or filtered by category and search keyword."""
        results = self.faqs
        if category and category != "all":
            results = [f for f in results if f.get("category") == category]

        if search:
            s = search.lower().strip()
            results = [
                f for f in results
                if s in f.get("question", "").lower()
                or s in f.get("answer", "").lower()
                or any(s in tag.lower() for tag in f.get("tags", []))
            ]
        return results

    def add_faq(
        self,
        question: str,
        answer: str,
        category: str = "custom",
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Dynamically adds a custom FAQ to dataset and re-fits TF-IDF live."""
        q = question.strip()
        a = answer.strip()
        if not q or not a:
            raise ValueError("Both 'question' and 'answer' are required.")

        new_id = f"faq_{int(time.time())}_{len(self.faqs) + 1}"
        new_entry = {
            "id": new_id,
            "category": category.strip() or "general",
            "question": q,
            "aliases": [q.lower()],
            "tags": [t.strip() for t in tags if t.strip()] if tags else [],
            "answer": a
        }

        self.faqs.append(new_entry)
        self._save_to_disk()
        self.load_and_fit()

        return {
            "status": "success",
            "message": "FAQ added and TF-IDF models re-fitted live!",
            "faq": new_entry,
            "total_faqs": len(self.faqs)
        }

    def edit_faq(
        self,
        faq_id: str,
        question: str,
        answer: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Updates an existing FAQ entry by ID and re-fits TF-IDF live."""
        q = question.strip()
        a = answer.strip()
        if not q or not a:
            raise ValueError("Both 'question' and 'answer' are required.")

        target = None
        for item in self.faqs:
            if str(item.get("id")) == str(faq_id):
                target = item
                break

        if not target:
            raise KeyError(f"FAQ with id '{faq_id}' not found.")

        target["question"] = q
        target["answer"] = a
        if category:
            target["category"] = category.strip()
        if tags is not None:
            target["tags"] = [t.strip() for t in tags if t.strip()]

        # Ensure question is in aliases
        if q.lower() not in target.get("aliases", []):
            target.setdefault("aliases", []).append(q.lower())

        self._save_to_disk()
        self.load_and_fit()

        return {
            "status": "success",
            "message": "FAQ updated and TF-IDF models re-fitted live!",
            "faq": target,
            "total_faqs": len(self.faqs)
        }

    def delete_faq(self, faq_id: str) -> Dict[str, Any]:
        """Deletes an FAQ entry by ID and re-fits TF-IDF live."""
        initial_len = len(self.faqs)
        self.faqs = [f for f in self.faqs if str(f.get("id")) != str(faq_id)]

        if len(self.faqs) == initial_len:
            raise KeyError(f"FAQ with id '{faq_id}' not found.")

        self._save_to_disk()
        self.load_and_fit()

        return {
            "status": "success",
            "message": f"FAQ '{faq_id}' removed and models re-fitted.",
            "total_faqs": len(self.faqs)
        }

    def _save_to_disk(self):
        """Persists current FAQ list to JSON file on disk."""
        self.faqs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.faqs_path, "w", encoding="utf-8") as f:
            json.dump(self.faqs, f, indent=2, ensure_ascii=False)

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on FAQ corpus, categories, and vectorizers."""
        categories: Dict[str, int] = {}
        for faq in self.faqs:
            cat = faq.get("category", "other")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_faqs": len(self.faqs),
            "categories": categories,
            "vocabulary_size": len(self.word_vectorizer.vocabulary_) if self.word_vectorizer else 0,
            "char_features": len(self.char_vectorizer.vocabulary_) if self.char_vectorizer else 0,
            "default_threshold": self.default_threshold
        }
