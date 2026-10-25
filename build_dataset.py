"""
build_dataset.py
Generates the comprehensive 128+ item dataset for CodeAlpha Hybrid AI Chatbot.
"""

import json
from pathlib import Path

faqs = []

def add(id, category, question, aliases, tags, answer):
    faqs.append({
        "id": id,
        "category": category,
        "question": question,
        "aliases": aliases,
        "tags": tags,
        "answer": answer.strip()
    })

# ==============================================================================
# 1. AI, MACHINE LEARNING & DATA SCIENCE (42 FAQs)
# ==============================================================================

add(
    "faq_ai_001",
    "ai_ml_data_science",
    "What is Artificial Intelligence (AI) and what are its main branches?",
    ["what is ai", "define ai", "artificial intelligence", "what does ai mean", "explain artificial intelligence", "branches of ai", "types of ai"],
    ["ai", "artificial intelligence", "machine learning", "deep learning", "nlp", "computer vision"],
    """### 🤖 What is Artificial Intelligence (AI)?

**Artificial Intelligence (AI)** is the branch of computer science dedicated to building systems capable of performing tasks that typically require human intelligence. These tasks include visual perception, speech recognition, decision-making, and language translation.

### Key Classifications:
1. **Narrow / Weak AI (ANI):** Specialized in a single domain (e.g., Siri, AlphaGo, recommendation engines).
2. **General AI (AGI):** Human-level intelligence across arbitrary domains (theoretical).
3. **Super AI (ASI):** Exceeding human capabilities across all disciplines.

### Core Subfields:
- **Machine Learning (ML):** Learning from data without explicit programming.
- **Deep Learning (DL):** Multi-layered artificial neural networks.
- **Natural Language Processing (NLP):** Understanding and generating human text and speech.
- **Computer Vision (CV):** Analyzing and interpreting visual data from the world."""
)

add(
    "faq_ai_002",
    "ai_ml_data_science",
    "What is the difference between AI, Machine Learning (ML), and Deep Learning (DL)?",
    ["difference between ml and dl", "ml vs dl", "ai vs ml vs dl", "difference between ai and ml", "compare ai ml and dl", "what is ml vs deep learning"],
    ["ai", "ml", "dl", "deep learning", "machine learning", "neural networks"],
    """### 🔄 AI vs. Machine Learning vs. Deep Learning

They form a nested hierarchy of technologies:

```text
┌────────────────────────────────────────────────────────┐
│  ARTIFICIAL INTELLIGENCE (Broadest Discipline)        │
│  ┌──────────────────────────────────────────────────┐  │
│  │  MACHINE LEARNING (Algorithms learning from data)│  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │  DEEP LEARNING (Multi-layer Neural Nets)   │  │  │
│  │  └────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

| Metric | Machine Learning (Traditional) | Deep Learning |
| :--- | :--- | :--- |
| **Feature Extraction** | Manual engineering (e.g., TF-IDF, SIFT) | Automatic hierarchical feature discovery |
| **Data Requirements** | Performs well on small-to-medium tabular sets | Requires massive datasets (millions of samples) |
| **Hardware** | Runs on standard CPUs | Requires GPU/TPU acceleration |
| **Algorithms** | Linear Regression, SVM, Random Forest, XGBoost | Transformers, CNNs, LSTMs, Diffusion Models |
| **Training Time** | Minutes to hours | Hours to weeks |"""
)

add(
    "faq_ai_003",
    "ai_ml_data_science",
    "What is Generative AI and how does it create new content?",
    ["what is generative ai", "explain genai", "what is gen ai", "generative ai vs discriminative ai", "how does genai work"],
    ["genai", "generative ai", "llm", "diffusion", "chatgpt", "gemini"],
    """### ✨ What is Generative AI?

**Generative AI (GenAI)** refers to deep learning models capable of generating novel, high-quality text, images, code, audio, and video that mirror human creativity.

### Discriminative vs. Generative:
- **Discriminative Models:** Model conditional probability $P(Y|X)$. They classify or predict labels (e.g., *'Is this email spam or not?'*).
- **Generative Models:** Model joint distribution $P(X, Y)$ or data distribution $P(X)$. They create new instances of data (e.g., *'Write an essay about quantum physics'*).

### Dominant Architectures:
1. **Autoregressive Transformers:** GPT-4, Google Gemini, Claude, LLaMA (Text & Code).
2. **Diffusion Models:** Stable Diffusion, Midjourney, DALL-E (Image & Video synthesis).
3. **Generative Adversarial Networks (GANs):** Generator vs. Discriminator game."""
)

add(
    "faq_ai_004",
    "ai_ml_data_science",
    "How do Large Language Models (LLMs) like ChatGPT and Gemini work under the hood?",
    ["how do llms work", "what is an llm", "how does chatgpt work", "how does gemini work", "explain large language models", "what is llm"],
    ["llm", "large language models", "chatgpt", "gemini", "transformers", "attention"],
    """### 🧠 How Large Language Models (LLMs) Work

LLMs are massive neural networks trained on vast text corpora using self-supervised objectives to predict the next token in a sequence.

### The 4-Stage Lifecycle:
1. **Pre-Training:** The model reads hundreds of billions of tokens (web, books, code) and minimizes cross-entropy loss by predicting masked or subsequent words.
2. **Instruction Fine-Tuning (SFT):** Trained on curated prompt-response pairs to follow instructions rather than simply completing text.
3. **Alignment (RLHF / DPO):** Reinforcement Learning from Human Feedback steers responses towards helpfulness, honesty, and safety.
4. **Inference (Sampling):** Uses temperature, top-$p$, and top-$k$ sampling to generate probabilities over vocabulary tokens sequentially."""
)

add(
    "faq_ai_005",
    "ai_ml_data_science",
    "What is Natural Language Processing (NLP) and what are its primary stages?",
    ["what is nlp", "define nlp", "natural language processing", "explain nlp", "nlp pipeline", "stages of nlp"],
    ["nlp", "natural language processing", "text processing", "tokenization", "lemmatization"],
    """### 🗣️ Natural Language Processing (NLP)

**NLP** bridges human communication and computational understanding.

### Complete NLP Processing Pipeline:
1. **Sentence Segmentation & Tokenization:** Breaking text into individual word tokens.
2. **Text Normalization:** Lowercasing, removing punctuation, converting accents.
3. **Stop-Word Removal:** Eliminating non-informative tokens (*'the'*, *'is'*, *'at'*).
4. **Lemmatization:** Converting words to their canonical dictionary root using POS tags.
5. **Feature Representation:** Bag-of-Words, TF-IDF, or dense contextual embeddings (Word2Vec, BERT).
6. **Downstream Inference:** Sentiment analysis, Named Entity Recognition (NER), Question Answering."""
)

add(
    "faq_ai_006",
    "ai_ml_data_science",
    "How does TF-IDF (Term Frequency-Inverse Document Frequency) work mathematically?",
    ["what is tfidf", "how does tf-idf work", "tf-idf formula", "explain tf idf", "term frequency inverse document frequency", "calculate tfidf"],
    ["tfidf", "term frequency", "information retrieval", "nlp", "scikit-learn"],
    """### 📊 TF-IDF Mathematical Formulation

**TF-IDF** scores the relevance of a word to a document relative to an entire corpus.

### Formulations:
1. **Term Frequency (TF):**
   $$\\text{TF}(t, d) = \\frac{\\text{Count of word } t \\text{ in document } d}{\\text{Total words in document } d}$$

2. **Inverse Document Frequency (IDF):** Penalizes universal words:
   $$\\text{IDF}(t, D) = \\ln\\left(\\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|}\\right) + 1$$

3. **Combined TF-IDF Weight:**
   $$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\text{IDF}(t, D)$$

Rare, highly informative terms (e.g., *'attendance'*, *'gradient'*) earn high weights; universal stop words receive near-zero weights."""
)

add(
    "faq_ai_007",
    "ai_ml_data_science",
    "What is Cosine Similarity and how does it calculate semantic distance between vectors?",
    ["what is cosine similarity", "cosine similarity formula", "how does cosine similarity work", "vector similarity", "semantic distance"],
    ["cosine similarity", "linear algebra", "vectors", "scikit-learn", "math"],
    """### 📐 Cosine Similarity in Vector Spaces

**Cosine Similarity** calculates the cosine of the angle between two non-zero vectors in an inner product space, disregarding their magnitude:

$$\\text{Cosine Similarity}(A, B) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|} = \\frac{\\sum_{i=1}^n A_i B_i}{\\sqrt{\\sum_{i=1}^n A_i^2} \\sqrt{\\sum_{i=1}^n B_i^2}}$$

### Properties:
- **Range:** `0.0` (orthogonal / no shared terms) to `1.0` (identical direction).
- **Length Invariant:** A short query and a detailed paragraph achieve high cosine similarity if they share proportional vocabulary weights."""
)

add(
    "faq_ai_008",
    "ai_ml_data_science",
    "What is an Artificial Neural Network (ANN) and how does forward propagation work?",
    ["what is a neural network", "explain ann", "artificial neural network", "how do neural networks work", "forward propagation", "perceptron"],
    ["neural network", "ann", "deep learning", "forward propagation", "weights"],
    """### 🕸️ Artificial Neural Networks (ANN)

An **Artificial Neural Network** is composed of interconnected layers of artificial neurons modeled after biological brains:

1. **Input Layer:** Receives numerical feature vector $X$.
2. **Hidden Layers:** Performs linear affine transformations followed by non-linear activations:
   $$z = W X + b$$
   $$a = \\sigma(z)$$
3. **Output Layer:** Produces predictions $\\hat{y}$ (probabilities via Softmax for classification, raw scalars for regression).
4. **Learning:** Errors are calculated via a loss function and propagated backward using **Backpropagation** to update weights $W$ via Gradient Descent."""
)

add(
    "faq_ai_009",
    "ai_ml_data_science",
    "How does a Convolutional Neural Network (CNN) work in Computer Vision?",
    ["what is a cnn", "explain cnn", "convolutional neural network", "how cnn works", "computer vision cnn"],
    ["cnn", "computer vision", "convolution", "pooling", "deep learning"],
    """### 🖼️ Convolutional Neural Networks (CNN)

CNNs preserve spatial topology in 2D image data using parameter sharing:

1. **Convolution Layer:** Slides learnable filters (kernels) across pixels to extract feature maps (edges, contours, textures).
2. **Activation (ReLU):** Applies $f(x) = \\max(0, x)$ to introduce non-linearity.
3. **Pooling Layer (Max Pooling):** Downsamples spatial dimensions ($2\\times 2$), reducing parameters and granting translation invariance.
4. **Dense / Fully Connected Layer:** Flattens high-level feature representations into classification probabilities."""
)

add(
    "faq_ai_010",
    "ai_ml_data_science",
    "What is a Recurrent Neural Network (RNN) and what problem do LSTMs solve?",
    ["what is rnn", "what is lstm", "rnn vs lstm", "recurrent neural network", "vanishing gradient problem"],
    ["rnn", "lstm", "sequential data", "vanishing gradient", "deep learning"],
    """### 🔁 RNNs & Long Short-Term Memory (LSTM)

- **Standard RNNs:** Process sequential inputs by feeding the previous hidden state $h_{t-1}$ into the current step $h_t$. However, backpropagation through long sequences triggers the **Vanishing Gradient Problem**, causing RNNs to forget early context.
- **LSTMs:** Introduce a constant-error carousel called the **Cell State** ($C_t$) regulated by three gates:
  1. **Forget Gate ($f_t$):** Discards obsolete information.
  2. **Input Gate ($i_t$):** Stores new incoming candidate values.
  3. **Output Gate ($o_t$):** Decides what reaches the hidden state."""
)

add(
    "faq_ai_011",
    "ai_ml_data_science",
    "What is the Transformer architecture and how does Self-Attention work?",
    ["what is a transformer", "how does transformer work", "self-attention mechanism", "attention is all you need", "transformer architecture"],
    ["transformer", "attention", "self-attention", "llm", "deep learning"],
    """### ⚡ The Transformer & Self-Attention

Introduced by Vaswani et al. (2017), Transformers replaced recurrence with **Scaled Dot-Product Attention**, allowing full parallelization on GPUs:

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

- **Query ($Q$):** What a token is searching for.
- **Key ($K$):** What a token contains / offers.
- **Value ($V$):** The contextual information passed through.
- **Multi-Head Attention:** Computes attention across multiple projection subspaces simultaneously, capturing syntactic and semantic dependencies."""
)

add(
    "faq_ai_012",
    "ai_ml_data_science",
    "What is Retrieval-Augmented Generation (RAG) and why is it essential?",
    ["what is rag", "retrieval augmented generation", "explain rag", "rag architecture", "why use rag"],
    ["rag", "vector database", "embeddings", "llm", "hallucination"],
    """### 📚 Retrieval-Augmented Generation (RAG)

**RAG** connects LLMs to verified external knowledge bases without model retraining:

1. **Ingest & Chunk:** Documents are split into semantic chunks and embedded into high-dimensional vector representations.
2. **Retrieve:** When a user asks a question, the vector database performs Approximate Nearest Neighbor (ANN) search for the most relevant context.
3. **Augment & Generate:** The user query and retrieved context are merged into an augmented prompt passed to the LLM.

### Key Benefits:
- Eliminates model hallucination.
- Provides verifiable citations.
- Instant knowledge base updates with zero GPU retraining."""
)

add(
    "faq_ai_013",
    "ai_ml_data_science",
    "What is Prompt Engineering and what are Zero-Shot, Few-Shot, and Chain-of-Thought?",
    ["what is prompt engineering", "zero-shot vs few-shot", "chain of thought prompting", "prompting techniques", "how to write prompts"],
    ["prompt engineering", "zero-shot", "few-shot", "cot", "llm"],
    """### 🎯 Prompt Engineering Techniques

Prompt engineering structures inputs to extract optimal outputs from LLMs:

- **Zero-Shot Prompting:** Asking the model directly without examples:
  *`"Classify sentiment: 'Delivery was two days early!'"`*
- **Few-Shot Prompting:** Supplying 2-4 exemplar input-output demonstrations before the target query to guide format and tone.
- **Chain-of-Thought (CoT):** Asking the model to show intermediate reasoning steps before concluding (*`"Think step-by-step before answering"`*), which drastically reduces math and logic errors."""
)

add(
    "faq_ai_014",
    "ai_ml_data_science",
    "What is AI Hallucination and how can it be prevented in production systems?",
    ["what is ai hallucination", "why do llms hallucinate", "how to prevent hallucination", "hallucination in ai", "grounding"],
    ["hallucination", "guardrails", "grounding", "rag", "reliability"],
    """### 🌫️ AI Hallucination & Mitigation

**AI Hallucination** occurs when a generative model outputs assertions that are factually incorrect or ungrounded while sounding completely confident.

### Prevention Strategies:
1. **RAG Grounding:** Force the model to answer exclusively from supplied source passages.
2. **Hybrid Verification:** Use deterministic local NLP (like our TF-IDF layer) for verified FAQs before calling generative fallbacks.
3. **Low Temperature:** Lower sampling temperature to `0.0` - `0.2` for deterministic responses.
4. **Explicit Refusal Guardrails:** Add system prompt instructions: *'If the answer is not contained in the context, state that you do not know.'*"""
)

add(
    "faq_ai_015",
    "ai_ml_data_science",
    "What is Overfitting and Underfitting, and how do you resolve them?",
    ["what is overfitting", "overfitting vs underfitting", "how to prevent overfitting", "bias variance tradeoff", "regularization"],
    ["overfitting", "underfitting", "regularization", "dropout", "validation"],
    """### ⚖️ Overfitting vs. Underfitting

- **Overfitting (High Variance):** Model memorizes training noise and outliers, performing poorly on test data.
- **Underfitting (High Bias):** Model is too simple to capture underlying patterns.

### How to Prevent Overfitting:
1. **Regularization:** Add L1 (Lasso) or L2 (Ridge) weight penalties.
2. **Cross-Validation:** Use 5-fold or 10-fold cross validation.
3. **Dropout:** In neural nets, randomly zero out 20-50% of activations during training.
4. **Early Stopping:** Stop training when validation loss stops decreasing.
5. **Data Augmentation:** Expand dataset with synthesized variations."""
)

add(
    "faq_ai_016",
    "ai_ml_data_science",
    "What is the difference between Supervised, Unsupervised, and Reinforcement Learning?",
    ["supervised vs unsupervised learning", "types of machine learning", "reinforcement learning vs supervised", "machine learning categories"],
    ["supervised", "unsupervised", "reinforcement learning", "clustering", "classification"],
    """### 🧪 The 3 Primary Paradigms of Machine Learning

1. **Supervised Learning:**
   - **Data:** Labeled pairs $(X, y)$.
   - **Goal:** Learn mapping function $f(X) \\approx y$.
   - **Examples:** Linear Regression, Random Forest, SVM, XGBoost.

2. **Unsupervised Learning:**
   - **Data:** Unlabeled input data $(X)$.
   - **Goal:** Discover latent structure, clusters, or distributions.
   - **Examples:** K-Means, PCA, Hierarchical Clustering, Autoencoders.

3. **Reinforcement Learning (RL):**
   - **Data:** Agent interacting with an Environment via States, Actions, and Rewards.
   - **Goal:** Maximize cumulative reward policy $\\pi(a|s)$.
   - **Examples:** Q-Learning, PPO, Deep Q-Networks (DQN)."""
)

add(
    "faq_ai_017",
    "ai_ml_data_science",
    "What is Reinforcement Learning from Human Feedback (RLHF)?",
    ["what is rlhf", "reinforcement learning from human feedback", "how chatgpt was aligned", "rlhf explained"],
    ["rlhf", "alignment", "ppo", "reward model", "llm"],
    """### 🤝 RLHF: Aligning LLMs with Human Intent

**RLHF** aligns language models to ensure they are helpful, harmless, and honest:

1. **Supervised Fine-Tuning (SFT):** Human annotators write ideal responses to prompts.
2. **Reward Model Training:** Annotators rank multiple model completions; a separate Reward Model is trained to predict human preference scores.
3. **PPO Optimization:** Proximal Policy Optimization (PPO) tunes the generative model to maximize reward scores while maintaining a KL-divergence penalty against drifting too far from the base model."""
)

add(
    "faq_ai_018",
    "ai_ml_data_science",
    "What is Computer Vision and what are object detection vs segmentation?",
    ["what is computer vision", "object detection vs segmentation", "image classification vs detection", "explain computer vision"],
    ["computer vision", "yolo", "segmentation", "object detection", "opencv"],
    """### 👁️ Computer Vision Hierarchy

Computer Vision enables computers to derive meaningful information from digital imagery:

1. **Image Classification:** Predicts a single class for the entire image (*'Cat'*).
2. **Object Detection:** Identifies multiple objects with bounding box coordinates $(x, y, w, h)$ and class labels (e.g., YOLO, Faster R-CNN).
3. **Semantic Segmentation:** Classifies every pixel into a category (e.g., road vs. sidewalk vs. car).
4. **Instance Segmentation:** Distinguishes between separate individual objects within the same class (Mask R-CNN)."""
)

add(
    "faq_ai_019",
    "ai_ml_data_science",
    "What is Tokenization in NLP (Byte-Pair Encoding, WordPiece)?",
    ["what is tokenization", "tokenization explained", "byte pair encoding", "bpe tokenization", "how tokenizer works"],
    ["tokenization", "bpe", "wordpiece", "nlp", "vocabulary"],
    """### 🧩 Tokenization in Modern NLP

**Tokenization** segments continuous raw strings into discrete tokens (words, subwords, or characters) mapped to integers in a vocabulary:

- **Word-Level:** Large vocabulary size; fails on out-of-vocabulary (OOV) misspellings.
- **Character-Level:** Small vocabulary; results in excessively long sequences.
- **Subword Tokenization (BPE / WordPiece / SentencePiece):**
  - Starts with characters and iteratively merges the most frequent pairs.
  - Frequent words remain whole (*'university'* $\\rightarrow$ `[university]`), while rare words decompose gracefully (*'unsupervised'* $\\rightarrow$ `[un, supervised]`).
  - Standard in modern LLMs (e.g., GPT uses tiktoken Byte-Pair Encoding)."""
)

add(
    "faq_ai_020",
    "ai_ml_data_science",
    "What is the difference between Stemming and Lemmatization in NLTK?",
    ["stemming vs lemmatization", "difference between stemming and lemmatization", "what is lemmatization", "what is stemming", "porter stemmer vs wordnet"],
    ["stemming", "lemmatization", "nltk", "spacy", "preprocessing"],
    """### ✂️ Stemming vs. Lemmatization

Both reduce inflected words to a base root:

- **Stemming (Rule-Based Heuristic):**
  - Chops prefixes and suffixes using crude algorithms (e.g., Porter Stemmer).
  - Fast, but often produces non-words (*'running'* $\\rightarrow$ *'run'*, but *'studies'* $\\rightarrow$ *'studi'*, *'better'* $\\rightarrow$ *'better'*).
- **Lemmatization (Lexical & Morphological Analysis):**
  - Consults a vocabulary dictionary (WordNet) and Part-of-Speech (POS) tags.
  - Always produces real dictionary words (*'studies'* $\\rightarrow$ *'study'*, *'better'* $\\rightarrow$ *'good'*).
  - **Verdict:** Lemmatization is strongly preferred for semantic search and question-answering engines."""
)

add(
    "faq_ai_021",
    "ai_ml_data_science",
    "What is a Confusion Matrix and how are Precision, Recall, and F1-Score calculated?",
    ["what is a confusion matrix", "precision vs recall", "f1-score formula", "how to calculate precision and recall", "classification metrics"],
    ["confusion matrix", "precision", "recall", "f1-score", "metrics"],
    """### 📊 Classification Evaluation Metrics

A **Confusion Matrix** visualizes prediction errors:

| Actual \\ Predicted | Predicted Positive | Predicted Negative |
| :--- | :--- | :--- |
| **Actual Positive** | **TP** (True Positive) | **FN** (False Negative) |
| **Actual Negative** | **FP** (False Positive) | **TN** (True Negative) |

### Key Formulas:
- **Precision:** $\\frac{\\text{TP}}{\\text{TP} + \\text{FP}}$ *(Of predicted positives, how many are real?)*
- **Recall (Sensitivity):** $\\frac{\\text{TP}}{\\text{TP} + \\text{FN}}$ *(Of actual positives, how many were caught?)*
- **F1-Score:** $2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$ *(Harmonic mean balancing both)*"""
)

add(
    "faq_ai_022",
    "ai_ml_data_science",
    "What is Gradient Descent and what is the role of Learning Rate?",
    ["what is gradient descent", "learning rate in machine learning", "how does gradient descent work", "stochastic gradient descent", "adam optimizer"],
    ["gradient descent", "optimization", "learning rate", "loss", "backpropagation"],
    """### 📉 Gradient Descent & Optimization

**Gradient Descent** iteratively adjusts model weights $\\theta$ in the opposite direction of the loss gradient $\\nabla J(\\theta)$ to find the minimum:

$$\\theta_{t+1} = \\theta_t - \\eta \\cdot \\nabla J(\\theta_t)$$

- **Learning Rate ($\\eta$):**
  - **Too large:** The optimizer overshoots the minimum and may diverge to $\\infty$.
  - **Too small:** Training is painfully slow and can get trapped in sub-optimal local minima.
- **Modern Optimizers:** **Adam (Adaptive Moment Estimation)** combines momentum and RMSprop to maintain per-parameter adaptive learning rates."""
)

# Continue adding AI items up to 42
ai_items = [
    ("faq_ai_023", "What is Cross-Entropy Loss?", ["what is cross entropy", "cross entropy loss formula", "log loss explained"], ["loss", "cross-entropy", "softmax"],
     "Cross-Entropy Loss (Log Loss) measures the difference between true categorical labels and predicted probability distributions. It heavily penalizes confident wrong predictions."),
    ("faq_ai_024", "What is the Bias-Variance Tradeoff?", ["what is bias variance tradeoff", "bias vs variance", "high bias high variance"], ["bias", "variance", "model complexity"],
     "Total Error = Bias² + Variance + Irreducible Error. High bias causes underfitting; high variance causes overfitting. The goal is to find the optimal model complexity that minimizes test error."),
    ("faq_ai_025", "What are the essential Python libraries for AI and Data Science?", ["python libraries for ai", "best python libraries for ml", "numpy pandas scikit-learn pytorch"], ["python", "numpy", "pandas", "pytorch", "scikit-learn"],
     "Key libraries include NumPy for matrix linear algebra, Pandas for DataFrame manipulation, scikit-learn for traditional ML & NLP vectorization, PyTorch/TensorFlow for deep learning, and NLTK/spaCy for linguistic preprocessing."),
    ("faq_ai_026", "What is a Vector Database (Chroma, Pinecone, FAISS)?", ["what is a vector database", "vector db explained", "faiss vs pinecone", "chromadb"], ["vector database", "embeddings", "faiss", "pinecone"],
     "Vector databases store high-dimensional dense embeddings and execute Approximate Nearest Neighbor (ANN) similarity lookups (like cosine or Euclidean distance) in sub-milliseconds across millions of records."),
    ("faq_ai_027", "What is Transfer Learning and Fine-Tuning in Deep Learning?", ["what is transfer learning", "fine tuning vs transfer learning", "what is finetuning"], ["transfer learning", "fine-tuning", "pretrained models"],
     "Transfer learning reuses weights from a large model pre-trained on generic data (e.g., ImageNet or web text) and adapts them to a specific niche task by fine-tuning on a smaller, domain-specific labeled dataset."),
    ("faq_ai_028", "What is Data Leakage and how do you avoid it?", ["what is data leakage", "data leakage in machine learning", "how to prevent data leakage"], ["data leakage", "train test split", "preprocessing"],
     "Data leakage occurs when information outside the training dataset is inadvertently used during model building. Prevent it by splitting train/test datasets BEFORE scaling, encoding, or TF-IDF vectorization."),
    ("faq_ai_029", "What is Principal Component Analysis (PCA)?", ["what is pca", "principal component analysis", "dimensionality reduction pca"], ["pca", "dimensionality reduction", "unsupervised"],
     "PCA is an unsupervised linear technique that finds orthogonal axes (principal components) maximizing data variance, compressing high-dimensional spaces into lower dimensions while minimizing information loss."),
    ("faq_ai_030", "What are Word Embeddings and how does Word2Vec work?", ["what is word2vec", "word embeddings explained", "cbow vs skip-gram"], ["word2vec", "embeddings", "nlp", "skipgram"],
     "Word embeddings map discrete words to continuous dense vectors where semantic relationships correspond to spatial arithmetic (e.g., Vector('King') - Vector('Man') + Vector('Woman') ≈ Vector('Queen')). Word2Vec uses CBOW or Skip-Gram."),
    ("faq_ai_031", "What is Backpropagation in neural networks?", ["what is backpropagation", "how does backpropagation work", "chain rule neural networks"], ["backpropagation", "chain rule", "gradients"],
     "Backpropagation calculates the gradient of the loss function with respect to every weight in the network by applying the calculus chain rule backwards from output to input layers."),
    ("faq_ai_032", "What are Activation Functions (ReLU, Sigmoid, Softmax)?", ["what is an activation function", "relu vs sigmoid", "softmax function explained"], ["activation function", "relu", "sigmoid", "softmax"],
     "Activation functions introduce non-linearity into neural networks, enabling them to learn complex non-linear boundaries. ReLU (max(0, x)) prevents vanishing gradients in hidden layers; Softmax outputs normalized probabilities in classification."),
    ("faq_ai_033", "What is K-Means Clustering and how does it choose centroids?", ["what is k-means", "how does kmeans work", "k means clustering algorithm"], ["kmeans", "clustering", "unsupervised"],
     "K-Means partitions N observations into K clusters. It initializes K centroids, assigns each point to the closest centroid via Euclidean distance, recalculates centroids as cluster means, and repeats until convergence."),
    ("faq_ai_034", "What is a Random Forest and how does Bagging work?", ["what is a random forest", "how random forest works", "bagging ensemble"], ["random forest", "decision tree", "bagging", "ensemble"],
     "Random Forest is an ensemble learning method that constructs a multitude of decision trees using Bootstrap Aggregation (Bagging) and random feature subsets. Aggregating their predictions drastically reduces overfitting."),
    ("faq_ai_035", "What is a Support Vector Machine (SVM)?", ["what is svm", "support vector machine explained", "kernel trick svm"], ["svm", "classification", "hyperplane", "kernel"],
     "SVM finds the optimal hyperplane that maximizes the margin of separation between distinct classes. The 'kernel trick' maps non-linearly separable inputs into higher-dimensional feature spaces where linear boundaries exist."),
    ("faq_ai_036", "What are Epoch, Batch Size, and Iteration?", ["epoch vs batch size vs iteration", "what is an epoch", "batch size explained"], ["epoch", "batch size", "iteration", "training"],
     "An Epoch is one complete forward and backward pass of the entire training dataset. Batch Size is the number of samples processed before updating weights. Iterations are the total batches required to complete one epoch."),
    ("faq_ai_037", "What is Data Augmentation in Computer Vision and NLP?", ["what is data augmentation", "data augmentation techniques", "image augmentation"], ["data augmentation", "cv", "nlp", "training"],
     "Data augmentation artificially expands training data by creating modified copies: rotations, flips, color jitter, and zooms for images; synonym replacement, back-translation, and random insertion for NLP."),
    ("faq_ai_038", "What is LangChain and how is it used in LLM app development?", ["what is langchain", "how to use langchain", "langchain explained"], ["langchain", "llm", "rag", "agents"],
     "LangChain is an open-source framework designed to simplify building applications powered by LLMs. It provides modular abstractions for chains, prompt templates, memory, document loaders, vector stores, and autonomous agents."),
    ("faq_ai_039", "What is Algorithmic Bias and Ethical AI?", ["what is algorithmic bias", "ethical ai explained", "fairness in machine learning"], ["ethics", "bias", "fairness", "safety"],
     "Algorithmic bias occurs when machine learning models produce systematically prejudiced decisions due to biased historical training data, unrepresentative sampling, or flawed model design. Ethical AI mandates transparency and fairness."),
    ("faq_ai_040", "How do you build a chatbot using NLP (Rule-Based vs Retrieval vs Generative)?", ["how to build a chatbot", "chatbot architectures", "retrieval vs generative chatbot"], ["chatbot", "nlp", "tfidf", "hybrid"],
     "Chatbots span 3 paradigms: Rule-based (pattern matching), Retrieval-based (TF-IDF vectorization and cosine similarity over an FAQ dataset, like Layer 1 of this bot!), and Generative (LLMs like Gemini/GPT, Layer 2 of this bot)."),
    ("faq_ai_041", "What is Sentiment Analysis in NLP?", ["what is sentiment analysis", "opinion mining", "sentiment classification nlp"], ["sentiment analysis", "nlp", "classification"],
     "Sentiment analysis determines the emotional polarity (Positive, Neutral, Negative) in subjective text using lexicon tools (VADER), classical TF-IDF classifiers, or pre-trained Transformer models (RoBERTa)."),
    ("faq_ai_042", "What is Named Entity Recognition (NER)?", ["what is ner", "named entity recognition", "extract entities nlp"], ["ner", "information extraction", "spacy", "nltk"],
     "Named Entity Recognition (NER) locates and classifies key information in unstructured text into predefined categories such as Person Names, Organizations, Locations, Time Expressions, and Monetary Values.")
]

for item in ai_items:
    add(item[0], "ai_ml_data_science", item[1], item[2], item[3], item[4])

# ==============================================================================
# 2. SOFTWARE ENGINEERING & WEB DEV (32 FAQs)
# ==============================================================================

add(
    "faq_se_001",
    "software_engineering_web",
    "What is Python and why is it the dominant language for AI and web development?",
    ["what is python", "why python for ai", "explain python language", "features of python", "is python interpreted"],
    ["python", "programming", "ai", "backend", "developer"],
    """### 🐍 Python: Features & Ecosystem

**Python** is a high-level, interpreted, dynamically typed programming language created by Guido van Rossum.

### Why Python Dominates AI & Web:
1. **Readable & Concise Syntax:** Code looks like pseudo-code, accelerating development cycles.
2. **Unrivaled ML/Data Ecosystem:** NumPy, Pandas, scikit-learn, PyTorch, and TensorFlow provide C-optimized mathematical backends.
3. **Versatile Web Frameworks:** Flask and FastAPI for lightweight microservices; Django for enterprise monoliths.
4. **Massive Community & Open Source Tooling:** Extensive package indexing via PyPI (`pip`)."""
)

add(
    "faq_se_002",
    "software_engineering_web",
    "Explain the 4 Pillars of Object-Oriented Programming (OOP) in Python.",
    ["what are the 4 pillars of oop", "oop concepts in python", "encapsulation inheritance polymorphism abstraction", "explain oop"],
    ["oop", "python", "encapsulation", "inheritance", "polymorphism", "abstraction"],
    """### 🏛️ The 4 Pillars of OOP in Python

1. **Encapsulation:** Bundling data (attributes) and methods that manipulate that data into a single class while restricting direct external access using private/protected conventions (`_var`, `__var`).
2. **Inheritance:** Enabling a child class to inherit methods and properties from a parent class, promoting code reuse:
   ```python
   class Dog(Animal):
       def speak(self): return "Woof!"
   ```
3. **Polymorphism:** Allowing different classes to define methods with the same name, behaving appropriately based on the calling object (Method Overriding / Duck Typing).
4. **Abstraction:** Hiding complex implementation details and exposing only the essential interface to the user using Python's `abc` module (`@abstractmethod`)."""
)

add(
    "faq_se_003",
    "software_engineering_web",
    "What is the difference between a Python List, Tuple, Set, and Dictionary?",
    ["list vs tuple vs dict", "difference between list and tuple", "python data structures", "list tuple set dictionary"],
    ["python", "list", "tuple", "dict", "set", "data structures"],
    """### 📊 Python Core Collections Comparison

| Data Structure | Syntax | Mutable? | Ordered? | Duplicates Allowed? | Common Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **List** | `[1, 2, 3]` | ✅ Yes | ✅ Yes | ✅ Yes | General ordered sequences |
| **Tuple** | `(1, 2, 3)` | ❌ No | ✅ Yes | ✅ Yes | Fixed records, dict keys |
| **Set** | `{1, 2, 3}` | ✅ Yes | ❌ No | ❌ No | Unique items, membership test |
| **Dictionary**| `{'a': 1}` | ✅ Yes | ✅ Yes (3.7+) | ❌ Unique keys | Key-Value mappings, fast lookup |"""
)

add(
    "faq_se_004",
    "software_engineering_web",
    "What is Flask and why is it preferred for building AI/NLP microservices?",
    ["what is flask", "why use flask", "flask for microservices", "flask web framework", "flask vs django"],
    ["flask", "python", "backend", "microservices", "web"],
    """### 🌶️ Flask Web Microframework

**Flask** is a lightweight Python WSGI web microframework designed for simplicity and modularity.

### Why Flask is ideal for AI / ML Microservices:
- **Zero Bloat:** Flask does not enforce monolithic database ORMs or administrative panels; you only import what you need.
- **In-Memory ML Model Loading:** Easily instantiates scikit-learn vectorizers, NLTK pipelines, or PyTorch models during server boot.
- **Minimal Routing:** Exposing RESTful API endpoints takes under 15 lines of clean code.
- **Production Pairing:** Pairs seamlessly with Gunicorn WSGI workers and Nginx reverse proxies for high concurrency."""
)

add(
    "faq_se_005",
    "software_engineering_web",
    "What is a RESTful API and how do HTTP status codes and methods work?",
    ["what is a rest api", "http methods explained", "http status codes", "restful api concepts", "get post put delete"],
    ["rest", "api", "http", "crud", "backend"],
    """### 🌐 REST APIs & HTTP Protocol

**REST (Representational State Transfer)** uses standard HTTP verbs for CRUD operations:

### HTTP Verbs:
- `GET`: Retrieve resource data without modifying server state.
- `POST`: Create a new resource with a payload.
- `PUT` / `PATCH`: Fully or partially update an existing resource.
- `DELETE`: Remove a resource.

### Essential HTTP Status Codes:
- `200 OK`: Successful retrieval / update.
- `201 Created`: Resource successfully created.
- `400 Bad Request`: Invalid client JSON payload.
- `401 Unauthorized`: Missing authentication credentials.
- `404 Not Found`: Target endpoint does not exist.
- `500 Internal Server Error`: Backend application crash."""
)

add(
    "faq_se_006",
    "software_engineering_web",
    "What is the difference between SQL (Relational) and NoSQL databases?",
    ["sql vs nosql", "difference between sql and nosql", "when to use nosql", "relational vs non relational database"],
    ["sql", "nosql", "database", "mongodb", "postgresql"],
    """### 🗄️ SQL vs. NoSQL Databases

| Feature | SQL (Relational) | NoSQL (Non-Relational) |
| :--- | :--- | :--- |
| **Schema** | Rigid, tabular, pre-defined columns | Dynamic, flexible schema (JSON/BSON, key-value) |
| **Data Integrity** | Strict ACID (Atomicity, Consistency, Isolation, Durability) | Eventual Consistency (BASE) |
| **Scaling** | Vertical scaling (Scale Up CPU/RAM) | Horizontal scaling (Scale Out across commodity nodes) |
| **Query Language** | Structured SQL queries (`SELECT... JOIN`) | Document queries, specialized APIs |
| **Examples** | PostgreSQL, MySQL, SQLite | MongoDB, Redis, Cassandra, DynamoDB |"""
)

add(
    "faq_se_007",
    "software_engineering_web",
    "What is the difference between Git and GitHub, and what are essential Git commands?",
    ["git vs github", "essential git commands", "difference between git and github", "git tutorial"],
    ["git", "github", "version control", "devops"],
    """### 🐙 Git vs. GitHub

- **Git:** A free, open-source distributed **version control CLI tool** installed locally to record changes to source code.
- **GitHub:** A cloud-hosted SaaS collaboration platform offering remote Git repositories, pull requests, issue tracking, and CI/CD pipelines.

### Essential Git Commands:
```bash
git init                  # Initialize local repository
git status                # Inspect modified & staged files
git add .                 # Stage all changes
git commit -m "feat: msg" # Save staged snapshot with message
git branch -M main        # Ensure branch name is main
git remote add origin <URL> # Link to GitHub remote
git push -u origin main   # Upload commits to GitHub
```"""
)

add(
    "faq_se_008",
    "software_engineering_web",
    "What is Big-O Notation and Time Complexity in Data Structures and Algorithms?",
    ["what is big o notation", "time complexity explained", "big o time complexities", "big o chart"],
    ["dsa", "big-o", "time complexity", "algorithms"],
    """### ⏱️ Big-O Computational Complexity

**Big-O Notation** describes the asymptotic upper bound of algorithm runtime or memory growth as input size $N \\to \\infty$:

| Big-O | Name | Example Algorithm |
| :--- | :--- | :--- |
| $O(1)$ | Constant Time | Hash Table lookup, accessing array index |
| $O(\\log N)$ | Logarithmic Time | Binary Search on sorted array |
| $O(N)$ | Linear Time | Linear Search, single loop over array |
| $O(N \\log N)$| Linearithmic | MergeSort, QuickSort (average case) |
| $O(N^2)$ | Quadratic Time | Nested loops, Bubble Sort, Insertion Sort |
| $O(2^N)$ | Exponential Time | Recursive Fibonacci without memoization |"""
)

# Additional Software Engineering Items
se_items = [
    ("faq_se_009", "How do you prepare for coding interviews and LeetCode?", ["how to prepare for coding interviews", "leetcode preparation roadmap", "crack software engineering interview"], ["interview", "dsa", "leetcode", "careers"],
     "Focus on high-yield patterns (Two Pointers, Sliding Window, Fast/Slow Pointers, BFS/DFS, Binary Search, Dynamic Programming). Practice 100-150 curated problems (NeetCode 150 / Blind 75) and write clean code while speaking your thoughts aloud."),
    ("faq_se_010", "What is Docker and how does Containerization differ from Virtual Machines?", ["what is docker", "docker vs vm", "containerization explained"], ["docker", "devops", "containers", "virtual machine"],
     "Docker packages apps and their dependencies into lightweight containers sharing the host OS kernel via Linux namespaces and cgroups. Unlike VMs with full guest OS layers, containers start in milliseconds and use minimal RAM."),
    ("faq_se_011", "What is CI/CD (Continuous Integration & Continuous Deployment)?", ["what is ci cd", "continuous integration explained", "github actions ci cd"], ["ci-cd", "devops", "github actions", "automation"],
     "CI automates building and unit testing whenever developers push code. CD automatically deploys validated builds to staging or production, enabling rapid and reliable software releases."),
    ("faq_se_012", "What is CORS (Cross-Origin Resource Sharing) and how to fix it in Flask?", ["what is cors", "how to fix cors in flask", "cross origin resource sharing"], ["cors", "flask", "security", "web"],
     "CORS is a browser security policy restricting web pages from making AJAX requests to a different domain/port. In Flask, resolve it using `flask_cors.CORS(app)` or by adding `Access-Control-Allow-Origin: *` response headers."),
    ("faq_se_013", "What is the difference between Synchronous and Asynchronous programming?", ["sync vs async", "difference between synchronous and asynchronous", "asyncio python"], ["async", "sync", "event loop", "python", "javascript"],
     "Synchronous code executes line-by-line, blocking execution during I/O operations (like database queries). Asynchronous code uses an event loop, allowing threads to handle other tasks while awaiting I/O responses."),
    ("faq_se_014", "What are Environment Variables and why are .env files used?", ["why use env files", "what are environment variables", "python dotenv explained"], ["security", "env", "api key", "dotenv"],
     "Environment variables keep sensitive secrets (API keys, DB credentials) out of source code. Never commit `.env` to Git; add it to `.gitignore` and distribute a sanitized `.env.example` file instead."),
    ("faq_se_015", "What is a Hash Table / Hash Map and how does it handle collisions?", ["what is a hash table", "hash map collision resolution", "hash table chaining"], ["hash table", "hash map", "collision", "dsa"],
     "A Hash Table maps keys to bucket indices using a hash function, providing $O(1)$ average time lookups. Collisions are handled via Chaining (linked lists at bucket) or Open Addressing (Linear Probing)."),
    ("faq_se_016", "What is a Linked List vs Array?", ["linked list vs array", "difference between array and linked list", "when to use linked list"], ["linked list", "array", "dsa", "memory"],
     "Arrays store contiguous memory, offering $O(1)$ index access but costly $O(N)$ arbitrary insertions. Linked Lists store nodes with data and pointers, allowing $O(1)$ insertions given the node pointer, but requiring $O(N)$ sequential traversal."),
    ("faq_se_017", "What is a Stack and Queue in Data Structures?", ["stack vs queue", "what is a stack", "what is a queue", "lifo vs fifo"], ["stack", "queue", "lifo", "fifo", "dsa"],
     "Stack is LIFO (Last-In, First-Out; `push` / `pop`), used in function call stacks and undo history. Queue is FIFO (First-In, First-Out; `enqueue` / `dequeue`), used in task scheduling and BFS graph traversal."),
    ("faq_se_018", "What is a Binary Search Tree (BST)?", ["what is a binary search tree", "bst explained", "binary tree vs bst"], ["bst", "tree", "binary search tree", "dsa"],
     "A BST is a node-based binary tree where every node's left subtree contains only values strictly less than the node, and the right subtree contains only values strictly greater, enabling $O(\\log N)$ average lookups."),
    ("faq_se_019", "What is Recursion and how does the Call Stack work?", ["what is recursion", "explain recursion", "base case recursion", "call stack recursion"], ["recursion", "call stack", "algorithms"],
     "Recursion is a programming technique where a function calls itself to solve smaller subproblems. Every recursion requires a Base Case to terminate and avoid Stack Overflow errors on the system call stack."),
    ("faq_se_020", "What is Binary Search and what is its precondition?", ["what is binary search", "how binary search works", "binary search condition"], ["binary search", "dsa", "algorithms", "log n"],
     "Binary Search repeatedly divides a search interval in half. Precondition: the collection MUST be sorted. Time complexity is $O(\\log N)$ compared to $O(N)$ for linear search."),
    ("faq_se_021", "What is MVC (Model-View-Controller) Architecture?", ["what is mvc", "mvc architecture explained", "model view controller"], ["mvc", "architecture", "design pattern"],
     "MVC decouples software into 3 components: Model (database schemas and business logic), View (user interface and presentation templates), and Controller (router handling HTTP input and dispatching actions)."),
    ("faq_se_022", "What are WebSockets and how do they differ from HTTP?", ["websockets vs http", "what is a websocket", "real time websockets"], ["websockets", "http", "realtime", "networking"],
     "HTTP is a stateless request-response protocol where clients poll servers. WebSockets establish a persistent, bidirectional, full-duplex TCP connection, ideal for live chats, stock tickers, and multiplayer gaming."),
    ("faq_se_023", "What is the difference between Frontend and Backend development?", ["frontend vs backend", "full stack developer explained"], ["frontend", "backend", "fullstack", "web"],
     "Frontend handles everything the user interacts with in the browser (HTML, CSS, JavaScript, React). Backend handles server architecture, APIs, authentication, databases, and business logic (Python, Flask, SQL, Node)."),
    ("faq_se_024", "What is JSON Web Token (JWT) and how does token authentication work?", ["what is jwt", "json web token explained", "jwt authentication"], ["jwt", "authentication", "security", "tokens"],
     "JWT is an open standard compact string containing Header, Payload, and Signature (`header.payload.signature`) used for stateless authentication without querying database sessions on every API request."),
    ("faq_se_025", "How do you optimize SQL database queries using Indexing?", ["how to optimize sql queries", "database indexing explained", "b tree index sql"], ["sql", "indexing", "database", "optimization"],
     "Indexes create $B$-tree data structures on selected table columns, speeding up `WHERE` and `JOIN` filters from $O(N)$ full table scans down to $O(\\log N)$ tree lookups, at the cost of slightly slower writes."),
    ("faq_se_026", "What is Unit Testing and how to write tests with pytest?", ["what is unit testing", "how to use pytest", "python pytest tutorial"], ["testing", "pytest", "unit test", "qa"],
     "Unit testing isolates and validates individual components or functions to prevent regression bugs. In Python, `pytest` executes automated test functions starting with `test_` using simple assertions."),
    ("faq_se_027", "How does Garbage Collection work in Python?", ["how does garbage collection work in python", "python memory management", "reference counting python"], ["python", "memory", "garbage collection"],
     "Python primarily uses Reference Counting (deallocating objects immediately when reference count reaches 0) paired with a cyclic Garbage Collector that detects and frees circular reference graphs in generations."),
    ("faq_se_028", "What is a Python Virtual Environment (venv) and why should you use it?", ["what is a virtual environment", "why use python venv", "create virtual environment python"], ["venv", "python", "virtualenv", "pip"],
     "A virtual environment isolates project dependencies, ensuring package version conflicts between projects are prevented and production `requirements.txt` remains clean and reproducible."),
    ("faq_se_029", "What is Monolithic vs Microservices architecture?", ["monolith vs microservices", "monolithic architecture explained", "microservices pros and cons"], ["architecture", "monolith", "microservices"],
     "Monolith combines UI, logic, and database into a single deployable unit. Microservices decompose functionality into independently deployable microservices communicating via HTTP/gRPC."),
    ("faq_se_030", "How does DNS and Web Browsing work when you type a URL?", ["how dns works", "what happens when you type a url", "dns lookup explained"], ["dns", "networking", "http", "browser"],
     "Browser checks local DNS cache, queries recursive DNS servers for the IP address, establishes a TCP 3-way handshake with TLS encryption, sends an HTTP GET request, and renders the returned HTML/CSS/JS."),
    ("faq_se_031", "What is Clean Code and PEP 8 in Python?", ["what is pep 8", "clean code python", "python style guide pep8"], ["pep8", "clean code", "python", "best practices"],
     "PEP 8 is Python's official style guide: 4 spaces per indentation level, snake_case for functions and variables, PascalCase for classes, imports at top of file, and 79-character line limits."),
    ("faq_se_032", "What is Agile and Scrum methodology?", ["what is agile", "what is scrum", "sprint standup agile"], ["agile", "scrum", "software engineering", "project management"],
     "Agile is an iterative software development approach emphasizing continuous delivery, customer collaboration, and adaptability. Scrum implements Agile using 2-week Sprints, Daily Standups, and Retrospectives.")
]

for item in se_items:
    add(item[0], "software_engineering_web", item[1], item[2], item[3], item[4])

# ==============================================================================
# 3. UNIVERSITY & CAMPUS LIFE (26 FAQs)
# ==============================================================================

uni_items = [
    ("faq_uni_001", "What are the eligibility criteria and admission process for B.Tech Computer Science?",
     ["eligibility for btech", "btech cse admission", "how to get admission in btech", "admission criteria cse", "btech cutoff"],
     ["admissions", "btech", "cse", "eligibility"],
     "Eligibility requires 60% aggregate in Class 12 with Physics, Mathematics, and Chemistry/Computer Science, plus a valid JEE Main or University Entrance exam rank. Apply online through the university portal."),
    
    ("faq_uni_002", "How does the Choice-Based Credit System (CBCS) and CGPA calculation work?",
     ["how is cgpa calculated", "sgpa to cgpa", "cbcs grading system", "credit system university", "cgpa formula"],
     ["cgpa", "sgpa", "cbcs", "credits", "grading"],
     "CBCS assigns credits to lecture, tutorial, and practical courses (160 credits required for graduation). SGPA is the weighted sum of grade points divided by semester credits; CGPA is the cumulative weighted mean across all semesters on a 10-point scale."),

    ("faq_uni_003", "What is the mandatory attendance policy and what happens if attendance falls below 75%?",
     ["what is attendance policy", "college attendance rules", "attendance below 75", "minimum attendance required", "medical leave attendance"],
     ["attendance", "rules", "exams", "medical leave"],
     "Students must maintain minimum 75% attendance in each registered course to write end-semester exams. Medical leave or official university duty can condone attendance up to 65% with approved certificates submitted within 5 days."),

    ("faq_uni_004", "What hostel facilities and dining mess options are available for students?",
     ["hostel facilities", "hostel timings and mess", "mess food quality", "hostel rules", "hostel accommodation"],
     ["hostel", "mess", "food", "rooms", "accommodation"],
     "Separate boys and girls hostels offer single, double, and triple AC/Non-AC rooms with Wi-Fi, study desks, and biometric security. Mess provides 4 hygienic meals daily with rotating menus. Curfew is 9:30 PM for freshers and 10:00 PM for seniors."),

    ("faq_uni_005", "What are the campus placement statistics, top recruiters, and average salary packages?",
     ["campus placement statistics", "average package cse", "top recruiters", "highest placement package", "placement criteria"],
     ["placements", "jobs", "careers", "salary", "package"],
     "Placement statistics: Highest package 44 LPA, Average package 7.5 to 9.2 LPA for B.Tech CSE. Top recruiters include Google, Microsoft, Amazon, TCS Digital, Infosys, Wipro, Accenture, and Deloitte. Minimum 6.5 CGPA required with no active backlogs."),

    ("faq_uni_006", "How do I apply for a summer internship and get an official No Objection Certificate (NOC)?",
     ["how to get noc for internship", "summer internship application", "internship approval university", "noc process"],
     ["internship", "noc", "training", "summer internship"],
     "Obtain your verified company offer letter, apply on the University ERP portal under Academic Services -> Request NOC, get HOD approval, and the official NOC will be issued within 3 working days. Mandatory internships carry 2 academic credits."),

    ("faq_uni_007", "What are the Central Library hours and borrowing rules for books and journals?",
     ["library hours", "how many books can i borrow", "library rules", "digital library access", "library timings"],
     ["library", "books", "borrowing", "ieee"],
     "Central Library is open Monday-Saturday 8:00 AM to 10:00 PM (open until midnight during exams). Undergrads can borrow 4 books for 14 days; free digital access is provided to IEEE Xplore, ScienceDirect, and ACM Digital Library on campus."),

    ("faq_uni_008", "What is the procedure for clearing backlogs and applying for supplementary re-examinations?",
     ["how to clear backlogs", "supplementary exam rules", "backlog re-exam procedure", "re-exam fee"],
     ["backlog", "re-exam", "supplementary", "exams"],
     "Supplementary exams are held prior to subsequent odd/even semesters. Register on ERP under Exam Registration within the notified deadline (fee ₹750/subject). Internal assessment marks are retained."),

    ("faq_uni_009", "What scholarships and financial assistance schemes are available for meritorious students?",
     ["scholarships available", "merit scholarship criteria", "fee waiver university", "financial aid students"],
     ["scholarship", "financial aid", "merit", "fee waiver"],
     "Merit scholarships provide up to 100% tuition waivers for top entrance rankers and 50% for students maintaining CGPA >= 9.5. Need-based concessions, defense category aid, and 15% girl-child scholarships are also available."),

    ("faq_uni_010", "What student clubs, coding societies, and technical hackathons are active on campus?",
     ["student clubs on campus", "coding clubs", "hackathons university", "gdsc acm club"],
     ["clubs", "societies", "hackathon", "gdsc", "acm"],
     "Active technical societies include Google Developer Student Clubs (GDSC), ACM Chapter, AI & Robotics Club, and CyberSecurity Club. Annual hackathons include CodeSpark and HackCampus with prizes up to ₹2,50,000."),

    ("faq_uni_011", "How do I pay university tuition fees and what is the policy on installment payments?",
     ["how to pay fees", "fee installment options", "tuition fee deadline", "late fee policy"],
     ["fees", "payment", "tuition", "installments"],
     "Pay fees online through the ERP student portal via Net Banking, UPI, or credit cards. Students facing economic constraints can request a 2-part semester installment plan with approval from the Accounts Officer."),

    ("faq_uni_012", "What campus transport and shuttle bus services are operated for day scholars?",
     ["campus transport", "bus routes day scholars", "college shuttle service", "bus pass"],
     ["transport", "bus", "shuttle", "commute"],
     "A fleet of 35+ GPS-enabled AC and non-AC buses covers all major metro stations and city hubs. Free continuous shuttles run every 15 minutes between the nearest metro station and the main campus gate."),

    ("faq_uni_013", "How do I connect to campus high-speed Wi-Fi and resolve network login issues?",
     ["connect to campus wifi", "wifi login portal", "wifi credentials roll number", "it helpdesk wifi"],
     ["wifi", "internet", "network", "helpdesk"],
     "Connect to SSID 'CampusNet-Student-Secure', open your browser, and log in with your Student Roll Number and ERP password. Maximum 2 registered devices per student. Visit Block-C Room 104 for IT support."),

    ("faq_uni_014", "What sports, fitness gymnasium, and recreational amenities are provided?",
     ["sports facilities", "gym timings", "cricket ground badminton court", "fitness gym"],
     ["sports", "gym", "fitness", "recreation"],
     "The campus features an indoor sports complex (badminton, table tennis, squash), floodlit football field, 400m running track, and modern fitness gym with trainers (6:00-8:30 AM & 4:30-9:00 PM)."),

    ("faq_uni_015", "What is the Anti-Ragging Policy and how can a student lodge a grievance safely?",
     ["anti-ragging policy", "ragging helpline number", "file grievance ragging", "anti ragging cell"],
     ["anti-ragging", "helpline", "grievance", "safety"],
     "Zero-tolerance anti-ragging policy with 24x7 toll-free helpline 1800-180-5522 and anonymous reporting boxes. Violations lead to immediate suspension, rustication, and filing of a police FIR."),

    ("faq_uni_016", "What healthcare facilities and medical centers are available on the campus?",
     ["medical center on campus", "health clinic doctor", "ambulance service college", "campus hospital"],
     ["medical", "health", "doctor", "ambulance"],
     "The Campus Health Center provides 24/7 basic medical care with resident physicians, emergency life-support ambulances, free essential medicines, and cashless tie-ups with city hospitals."),

    ("faq_uni_017", "How do I get my student ID card replaced if lost or damaged?",
     ["replace lost id card", "duplicate id card college", "lost student id"],
     ["id card", "replacement", "security"],
     "Report the lost card to Campus Security to deactivate RFID access, pay ₹250 replacement fee on ERP, and submit the receipt with a photo to the Student Affairs Office to receive a new card in 48 hours."),

    ("faq_uni_018", "What specialization electives are offered in B.Tech Computer Science?",
     ["btech cse electives", "cse specialization tracks", "ai cloud cybersecurity tracks"],
     ["specialization", "electives", "cse", "tracks"],
     "Specialization tracks starting 5th semester: 1) Artificial Intelligence & Machine Learning, 2) Cloud Computing & DevOps, 3) Cyber Security & Digital Forensics, and 4) Data Science & Big Data."),

    ("faq_uni_019", "What is the procedure for semester registration and elective course selection?",
     ["semester registration", "how to choose electives", "course registration erp"],
     ["registration", "courses", "electives", "erp"],
     "Dates are announced 2 weeks prior to semester start. Log in to ERP -> Academic -> Course Registration, select program and open electives, and get faculty mentor sign-off to finalize."),

    ("faq_uni_020", "Where can I view my semester exam results and request grade re-evaluation?",
     ["check exam results", "re-evaluation process", "marksheet marks verification"],
     ["results", "grades", "revaluation", "transcripts"],
     "View results on the University Examination Portal using your Hall Ticket number. Apply for answer sheet photocopy or formal re-evaluation within 10 days of results (fee ₹500/paper)."),

    ("faq_uni_021", "What mental health and psychological counseling services does the university provide?",
     ["counseling center", "mental health support", "stress management therapy", "wellness counselor"],
     ["counseling", "mental health", "wellness", "therapy"],
     "Confidential 1-on-1 sessions with licensed clinical psychologists for exam anxiety, stress, and interpersonal wellness in Block-B Room 202. Completely private and free."),

    ("faq_uni_022", "How can students participate in university research labs and publish papers?",
     ["research labs undergraduate", "publish research paper ieee", "student research grants"],
     ["research", "publications", "ieee", "projects"],
     "Connect with faculty mentors during 2nd and 3rd year. University provides up to ₹50,000 seed grants for student patent prototypes and travel reimbursement for IEEE/ACM Scopus conferences."),

    ("faq_uni_023", "What are the best final year project ideas for B.Tech CSE students?",
     ["final year project ideas", "btech cse project ideas", "ai project ideas for college", "major project topics"],
     ["projects", "btech", "final year", "capstone"],
     "Top project ideas: 1) Dual-Layer Hybrid NLP Chatbot (like this project!), 2) Real-Time YOLO Object Detection & Tracking, 3) AI Healthcare Disease Prediction, 4) Blockchain Credential Verification."),

    ("faq_uni_024", "How do I build an impressive resume for tech campus placements?",
     ["resume tips for tech placements", "how to build tech resume", "software engineer resume guide"],
     ["resume", "placements", "career", "interview"],
     "Use a clean single-page template (Jake's Resume / LaTeX), lead with technical skills (Languages, Frameworks, Tools), detail 2-3 GitHub-backed projects with live demo links, and quantify achievements with metrics."),

    ("faq_uni_025", "How can I secure an off-campus software or AI internship?",
     ["how to get off-campus internship", "off campus tech internship guide", "cold emailing for internships"],
     ["internship", "off-campus", "jobs", "careers"],
     "Build 2-3 polished full-stack or AI projects on GitHub, solve 100+ LeetCode problems, optimize your LinkedIn profile, and reach out to engineering managers with concise personalized cold emails."),

    ("faq_uni_026", "What is the university academic calendar and holiday schedule?",
     ["academic calendar", "semester break dates", "exam schedule holidays"],
     ["calendar", "holidays", "semester break", "schedule"],
     "Odd semesters run from August to December; even semesters run from January to May. Winter break is 3 weeks in December-January; summer internship break spans June and July.")
]

for item in uni_items:
    add(item[0], "university_campus", item[1], item[2], item[3], item[4])

# ==============================================================================
# 4. CONVERSATIONAL & GENERAL ASSISTANT (28 FAQs)
# ==============================================================================

add(
    "faq_chat_001",
    "conversational_smalltalk",
    "Hello! Hi there, how are you?",
    ["hi", "hello", "hey", "how are you", "good morning", "good evening", "greetings", "hi there"],
    ["greeting", "hello", "hi", "salutation"],
    """Hello! 👋 I am your **Hybrid AI FAQ & General Assistant**, built for the **CodeAlpha AI Internship**.

I can assist you with:
- 🏫 **University & Campus Life** (Admissions, curriculum, hostels, placements)
- 🤖 **AI, Machine Learning & NLP** (TF-IDF, Cosine Similarity, Deep Learning)
- 💻 **Software Engineering & Web Dev** (Python, Flask, Git, Databases)
- 🧮 **Live Math, Logic & Real-Time Utilities**
- 🐍 **Writing Working Python Code Snippets**

How can I help you today? Ask me any question or pick a starter prompt from the sidebar!"""
)

add(
    "faq_chat_002",
    "conversational_smalltalk",
    "Who created you and what is your purpose?",
    ["who are you", "who created you", "who made you", "what is your purpose", "what do you do", "codealpha chatbot"],
    ["creator", "identity", "codealpha", "purpose"],
    """I was created as a capstone project for the **CodeAlpha Artificial Intelligence Internship**! 🎓

### My Purpose:
To demonstrate a **production-grade Hybrid NLP Architecture** that combines deterministic local text retrieval (via **NLTK Lemmatization**, **TF-IDF Vectorization**, and **Cosine Similarity**) with intelligent **Generative AI fallback routing** (via Google Gemini API / built-in offline smart heuristics). This ensures fast, reliable, explainable, and zero-downtime assistance!"""
)

add(
    "faq_chat_003",
    "conversational_smalltalk",
    "What can you do and what features do you support?",
    ["what can you do", "features of this bot", "tell me your capabilities", "what are your skills"],
    ["features", "capabilities", "skills", "help"],
    """### 🚀 Key Capabilities & Features:

1. **Dual-Layer Hybrid Answering:** Matches verified campus and tech FAQs locally using TF-IDF and Cosine Similarity, or routes to Gemini GenAI fallback.
2. **Live Math & Expression Solver:** Computes natural language expressions (`"2 times 4"`, `"150 + 350 / 5"`, `"square root of 144"`).
3. **Real-Time Utilities:** Tells today's date, current time, coin flips, and word counts.
4. **Python Code Generator:** Provides tested, working implementations for algorithms (Fibonacci, Prime, Palindrome, Binary Search).
5. **Explainable AI Metadata Badges:** Displays exact similarity confidence %, latency, and matched FAQ tags on every response.
6. **Live FAQ Retraining:** Add custom Q&A pairs via the web modal or `/add-faq` API with instant in-memory TF-IDF vectorizer re-fitting!"""
)

add(
    "faq_chat_004",
    "conversational_smalltalk",
    "Tell me a programming joke!",
    ["tell me a joke", "programming joke", "funny coding joke", "developer joke", "make me laugh"],
    ["joke", "humor", "fun", "programming"],
    """Here are two classic developer jokes for you! 😄

> **Why do programmers prefer dark mode?**
> *Because light attracts bugs!* 🐛

> **There are 10 types of people in the world:**
> *Those who understand binary, and those who don't.* 💻

> **A SQL query walks into a bar, walks up to two tables and asks:**
> *"Can I join you?"* 🍺"""
)

add(
    "faq_chat_005",
    "conversational_smalltalk",
    "Give me an inspiring motivational quote for coding and engineering.",
    ["motivational quote", "inspire me", "quote for coders", "motivation for studying"],
    ["quote", "motivation", "inspiration", "study"],
    """Here is some inspiration for your engineering journey: 💡

> *"The only way to do great work is to love what you do."* — Steve Jobs
>
> *"Any fool can write code that a computer can understand. Good programmers write code that humans can understand."* — Martin Fowler
>
> *"It's not that I'm so smart, it's just that I stay with problems longer."* — Albert Einstein

Stay persistent, keep building real projects, and remember that every bug you fix makes you a stronger engineer! 🚀"""
)

add(
    "faq_chat_006",
    "conversational_smalltalk",
    "How to study effectively and master technical subjects?",
    ["how to study effectively", "study tips for engineering", "pomodoro technique", "feynman technique"],
    ["study", "productivity", "pomodoro", "feynman", "tips"],
    """### 🎓 5 Proven Study Techniques for Engineers:

1. **The Feynman Technique:** Explain complex algorithms in simple plain English without jargon. Wherever you struggle to explain simply is where your knowledge gap lies.
2. **Active Recall & Spaced Repetition:** Test yourself with flashcards (Anki) rather than passively re-reading slides.
3. **Pomodoro Technique:** 25 minutes of 100% focused work with zero distractions, followed by a 5-minute break.
4. **Build Real Code:** Learn by implementing algorithms from scratch, making errors, and debugging them.
5. **Prioritize 7+ Hours Sleep:** Deep REM sleep consolidates neural pathways required for memory retention and problem-solving."""
)

add(
    "faq_chat_007",
    "conversational_smalltalk",
    "Write a Python program for the Fibonacci series.",
    ["write a python program for fibonacci", "fibonacci series in python", "python code for fibonacci", "fibonacci sequence code"],
    ["python", "code", "fibonacci", "algorithms"],
    """Here is a complete, optimized Python program to generate the **Fibonacci Series**:

```python
def fibonacci(n: int) -> list[int]:
    \"\"\"Generate Fibonacci sequence up to n terms.\"\"\"
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    sequence = [0, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence

# Example usage:
n_terms = 10
print(f"Fibonacci series ({n_terms} terms): {fibonacci(n_terms)}")
# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

### Recursive with Memoization:
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_memo(n: int) -> int:
    if n < 2:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)

print(fib_memo(10))  # Output: 55
```"""
)

add(
    "faq_chat_008",
    "conversational_smalltalk",
    "Write a Python program to check if a number is Prime.",
    ["write a python program for prime number", "check prime number in python", "python code prime number", "is prime python"],
    ["python", "code", "prime", "math"],
    """Here is an efficient $O(\\sqrt{N})$ Python function to check for **Prime Numbers**:

```python
import math

def is_prime(n: int) -> bool:
    \"\"\"Return True if n is a prime number, else False.\"\"\"
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    # Check potential factors up to sqrt(n) with 6k +/- 1 optimization
    for i in range(5, int(math.isqrt(n)) + 1, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True

# Example test cases:
for num in [2, 17, 25, 29, 97, 100]:
    print(f"{num}: {'Prime' if is_prime(num) else 'Not Prime'}")
```"""
)

add(
    "faq_chat_009",
    "conversational_smalltalk",
    "Write a Python program to check if a string is a Palindrome.",
    ["write a python program for palindrome", "check palindrome in python", "palindrome python code", "is palindrome string"],
    ["python", "code", "palindrome", "strings"],
    """Here is a clean Python program to check if a string or number is a **Palindrome**:

```python
def is_palindrome(text: str) -> bool:
    \"\"\"Check if text reads the same forwards and backwards (ignoring case and non-alphanumeric).\"\"\"
    # Clean text: lowercase and alphanumeric only
    cleaned = [ch.lower() for ch in str(text) if ch.isalnum()]
    return cleaned == cleaned[::-1]

# Example tests:
print(is_palindrome("racecar"))                  # True
print(is_palindrome("A man, a plan, a canal: Panama")) # True
print(is_palindrome("hello world"))              # False
print(is_palindrome(12321))                      # True
```"""
)

add(
    "faq_chat_010",
    "conversational_smalltalk",
    "Write a Python program to calculate the Factorial of a number.",
    ["write a python program for factorial", "factorial in python", "python code factorial", "calculate factorial python"],
    ["python", "code", "factorial", "math"],
    """Here are iterative and recursive implementations of **Factorial** in Python:

```python
# 1. Iterative Approach - O(N) Time, O(1) Space
def factorial_iterative(n: int) -> int:
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# 2. Recursive Approach
def factorial_recursive(n: int) -> int:
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n in (0, 1):
        return 1
    return n * factorial_recursive(n - 1)

print(factorial_iterative(5))  # Output: 120
print(factorial_recursive(6))  # Output: 720
```"""
)

add(
    "faq_chat_011",
    "conversational_smalltalk",
    "Write a Python program for Binary Search on a sorted array.",
    ["write a python program for binary search", "binary search in python", "python code binary search", "binary search algorithm code"],
    ["python", "code", "binary search", "dsa"],
    """Here is the standard iterative **Binary Search** algorithm in Python with $O(\\log N)$ time complexity:

```python
def binary_search(arr: list[int], target: int) -> int:
    \"\"\"Return the index of target in sorted arr, or -1 if not found.\"\"\"
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Prevents integer overflow
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return -1

# Example test:
numbers = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
target_val = 23
idx = binary_search(numbers, target_val)
print(f"Target {target_val} found at index: {idx}")  # Output: index 5
```"""
)

add(
    "faq_chat_012",
    "conversational_smalltalk",
    "Write a Python program for Bubble Sort and QuickSort.",
    ["write a python program for bubble sort", "quicksort in python", "sorting algorithms python", "python code for quick sort"],
    ["python", "code", "quicksort", "bubblesort", "sorting"],
    """Here are implementations of **Bubble Sort** and **QuickSort** in Python:

```python
# 1. QuickSort - Average O(N log N)
def quicksort(arr: list) -> list:
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# 2. Bubble Sort - O(N^2)
def bubble_sort(arr: list) -> list:
    a = arr.copy()
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a

test_list = [64, 34, 25, 12, 22, 11, 90]
print("QuickSorted:", quicksort(test_list))
print("BubbleSorted:", bubble_sort(test_list))
```"""
)

# Other Conversational items
chat_items = [
    ("faq_chat_013", "How do I learn Python from scratch?", ["how to learn python", "python beginner roadmap", "learn python programming"], ["python", "learning", "roadmap"],
     "Phase 1: Basics (syntax, data structures, loops, functions). Phase 2: OOP & modules. Phase 3: Packages (NumPy, Pandas, Flask). Phase 4: Build real projects and push code to GitHub."),
    ("faq_chat_014", "How to contribute to open-source software on GitHub?", ["how to contribute to open source", "github open source guide", "first pull request"], ["open-source", "github", "git", "pr"],
     "Find repositories with 'good first issue' labels, fork the repo, create a dedicated feature branch, write clean tested code adhering to project guidelines, and open a concise Pull Request."),
    ("faq_chat_015", "Thank you! Goodbye!", ["thank you", "thanks", "bye", "goodbye", "see you later"], ["thanks", "bye", "farewell"],
     "You are very welcome! 😊 It was a pleasure assisting you. Good luck with your studies, coding projects, and internship journey! Feel free to come back whenever you have questions."),
    ("faq_chat_016", "How do I add a custom FAQ to this chatbot?", ["how to add custom faq", "add question to dataset", "how to train bot"], ["add-faq", "custom", "retrain"],
     "Click '+ Add Custom FAQ' in the header toolbar, enter the Question, Category, Tags, and Markdown Answer, and click Save. The TF-IDF vectorizer will re-fit live in memory!"),
    ("faq_chat_017", "What should I do if the bot gives an inaccurate answer?", ["inaccurate answer", "bot gave wrong answer", "explainable ai debug"], ["feedback", "accuracy", "explainable-ai"],
     "Check the confidence pill badge below the answer. You can raise the threshold slider in the sidebar to enforce stricter matching, toggle into Strict FAQ Mode, or add the exact question via '+ Add Custom FAQ'."),
    ("faq_chat_018", "What is CodeAlpha?", ["what is codealpha", "about codealpha", "codealpha internship"], ["codealpha", "internship", "organization"],
     "CodeAlpha is an education and technology career platform offering hands-on virtual internships in Artificial Intelligence, Machine Learning, Web Development, and Cybersecurity to empower student developers with practical project experience."),
    ("faq_chat_019", "Can you explain how this chatbot project was built?", ["how was this chatbot built", "chatbot architecture explanation", "project details"], ["architecture", "internship", "flask", "nlp"],
     "Built using Python, Flask, scikit-learn, NLTK, HTML5, CSS3 glassmorphism, and Vanilla JS. It features a dual-layer NLP engine (TF-IDF + Cosine Similarity primary, Gemini GenAI fallback, and zero-key offline heuristics)."),
    ("faq_chat_020", "Tell me a fun tech fact!", ["fun tech fact", "interesting technology fact", "tell me a fact"], ["fun", "fact", "trivia"],
     "Did you know? The first computer bug was an actual real moth found trapped inside a relay of the Harvard Mark II computer in 1947 by Grace Hopper's team!"),
    ("faq_chat_021", "What are good tech podcasts or YouTube channels?", ["best tech podcasts", "youtube channels for coding", "recommend tech channels"], ["learning", "resources", "podcasts"],
     "Top YouTube channels: 3Blue1Brown (Math/Neural Networks), freeCodeCamp, Fireship (tech overviews), StatQuest with Josh Starmer (ML/Statistics), and Sentdex (Python AI). Podcasts: Lex Fridman Podcast and Syntax.fm."),
    ("faq_chat_022", "How do I manage stress during college exam season?", ["exam stress management", "handling exam anxiety", "study stress"], ["wellness", "stress", "mental health"],
     "Break study sessions into 25-minute Pomodoro intervals, take brisk 15-minute walks outside, stay hydrated, maintain 7+ hours of sleep, and avoid cramming new difficult topics the night before exams."),
    ("faq_chat_023", "Why is Pair Programming useful in software teams?", ["why pair programming", "benefits of pair programming", "pair programming explained"], ["pair-programming", "software-engineering", "teamwork"],
     "Pair programming (Driver + Navigator) reduces bugs by up to 50%, facilitates continuous real-time code review, spreads domain knowledge across the team, and accelerates onboarding for junior developers."),
    ("faq_chat_024", "How to debug code effectively when stuck?", ["how to debug code", "debugging tips", "rubber duck debugging"], ["debugging", "troubleshooting", "clean-code"],
     "1) Read the entire traceback line-by-line; 2) Use Rubber Duck Debugging (explain the code aloud to an inanimate object); 3) Isolate the bug with minimal reproducible examples; 4) Use debuggers (`pdb` / breakpoints) rather than endless print statements."),
    ("faq_chat_025", "What is an API Key and how to protect it?", ["what is an api key", "how to protect api keys", "api key security"], ["security", "api key", "credentials"],
     "An API Key is a unique token used to authenticate an application to an API service. Protect it by storing it exclusively in `.env` files, omitting `.env` from Git commits, and setting usage budget limits."),
    ("faq_chat_026", "What is the difference between a Compiler and an Interpreter?", ["compiler vs interpreter", "compiled vs interpreted language"], ["compiler", "interpreter", "programming"],
     "A Compiler translates source code entirely into machine binary before execution (C, C++, Rust - high speed). An Interpreter executes code line-by-line in real time (Python, Ruby - high flexibility and rapid testing)."),
    ("faq_chat_027", "What is Markdown and why is it used everywhere?", ["what is markdown", "markdown guide", "why use markdown"], ["markdown", "documentation", "readme"],
     "Markdown is a lightweight markup language that formats plain text into styled HTML using simple symbols (`#` for headers, `**` for bold, ```` ``` ```` for code blocks), making it the universal standard for GitHub READMEs."),
    ("faq_chat_028", "Farewell! Have a great day!", ["bye bye", "farewell", "have a great day", "see you"], ["farewell", "bye", "goodbye"],
     "Goodbye! Have a fantastic day ahead, and keep coding! Reach out whenever you need assistance! 🚀")
]

for item in chat_items:
    add(item[0], "conversational_smalltalk", item[1], item[2], item[3], item[4])

# Write out to data/faqs.json
target_file = Path("data/faqs.json")
target_file.parent.mkdir(parents=True, exist_ok=True)
with open(target_file, "w", encoding="utf-8") as f:
    json.dump(faqs, f, indent=2, ensure_ascii=False)

print(f"Successfully generated {len(faqs)} FAQs in {target_file}")
