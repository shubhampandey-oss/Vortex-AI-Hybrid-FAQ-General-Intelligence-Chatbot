# ⚡ Vortex AI — Hybrid FAQ & General Intelligence Chatbot

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask 3.1](https://img.shields.io/badge/Flask-3.1-000000.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Google GenAI](https://img.shields.io/badge/Google%20GenAI-Gemini%20API-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991.svg?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7.svg?logo=render&logoColor=white)](https://render.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

> **Vortex AI (`vortex.ai`)** is a production-grade, typography-first Hybrid AI Chatbot built with **Swiss International Typographic Style × Luxury Brutalism**. It unifies deterministic local NLP retrieval (**TF-IDF Cosine Similarity** over a verified 128+ FAQ knowledge base) with neural **Generative AI Assistant routing** (**Google Gemini 3.5** and **OpenAI GPT-4o-mini**) for deep reasoning, coding, and multi-turn conversational synthesis.

---

## 📸 Key Features & Capabilities

- **Dual-Engine Hybrid Intelligence:**
  - **Deterministic Retrieval:** Dual-Channel TF-IDF vectorizer (Word n-grams `(1, 2)` + Character-wb n-grams `(3, 5)` for typo resilience) with cosine proximity scoring over 128+ categorized FAQs.
  - **Neural Synthesis:** Autonomous routing to Google Gemini (`gemini-3.5-flash`) or OpenAI with full multi-turn conversational context retention.
- **Three Intelligent Routing Modes:**
  1. `AUTO` *(Default)*: Evaluates verified local FAQ first. If confidence $\ge$ threshold, returns verified answer; otherwise routes dynamically to Generative AI.
  2. `FAQ ONLY`: Direct queries strictly against the local knowledge base. External LLMs are never invoked.
  3. `AI ASSISTANT`: Direct neural reasoning for complex questions, code generation, debugging, and open-domain problem solving.
- **Dynamic Knowledge Base CRUD (`/faqs`, `/add-faq`, `/edit-faq`, `/delete-faq`):**
  - Live administrative modal to search, browse, add, edit, or remove FAQs with instant live vectorizer re-fitting in memory (zero server downtime).
- **Swiss Minimalism × Monochromatic Editorial Design:**
  - Responsive 5-layer typographic echo stack for `VORTEX`.
  - Clash Display (700) headlines, Satoshi (500) body typography, and JetBrains Mono code blocks.
  - High-contrast monochromatic light (`#F2F2F2`) and dark (`#111111`) themes with instant toggle and `localStorage` persistence.
  - Collapsible history sidebar with real-time search, session deletion, and conversation export (TXT / JSON).
- **Zero-Key Offline Resilience:**
  - AST-safe Math solver and utility engines ensure the application remains fully functional even without external API keys.

---

## 📁 Project Structure

```
├── app.py                 # Flask server, REST API endpoints & route handlers
├── config.py              # Central configuration & environment loader
├── nlp_engine.py          # Dual-layer hybrid controller & query dispatcher
├── faq_engine.py          # TF-IDF Cosine Similarity vectorizer & dynamic CRUD
├── llm_engine.py          # Google Gemini & OpenAI LLM integrations
├── math_engine.py         # AST safe mathematical evaluation engine
├── utility_engine.py      # Real-time clock, calendar, and system utilities
├── test_chatbot.py        # Comprehensive 12-test unit verification suite
├── test_accuracy.py       # Benchmark evaluation script (100% accuracy)
├── build_dataset.py       # Knowledge base generator script
├── data/
│   └── faqs.json          # 128+ verified FAQ dataset
├── templates/
│   └── index.html         # Swiss editorial chat UI template
├── static/
│   ├── css/
│   │   └── style.css      # Monochromatic design system tokens & responsive rules
│   └── js/
│       └── app.js         # Client-side session management & chat logic
├── requirements.txt       # Production Python dependencies (includes gunicorn)
├── render.yaml            # Render Blueprint deployment configuration
├── Procfile               # Production WSGI process declaration
├── runtime.txt            # Python runtime version declaration (3.11.9)
├── .env.example           # Environment variable template
└── .gitignore             # Secrets & cache ignore rules
```

---

## 📤 Step 1: Uploading to GitHub

1. **Verify Git Status**
   Ensure `.env` is **never** committed:
   ```bash
   git status
   ```
2. **Commit All Files**
   ```bash
   git add .
   git commit -m "Initial release of Vortex AI Hybrid Chatbot"
   ```
3. **Create a New Repository on GitHub**
   - Go to [github.com/new](https://github.com/new).
   - Name your repository (e.g., `vortex-ai-chatbot`).
   - Choose **Public** or **Private**.
   - Do **not** check "Initialize with README" or .gitignore (we already have them).
4. **Push to GitHub**
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

---

## 🌐 Step 2: Deploying to Render

You can deploy to [Render](https://render.com) using either **Method A (Blueprint - Easiest)** or **Method B (Manual)**.

### Method A: 1-Click Blueprint Deployment (Recommended)

1. Sign in to your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** in the top navigation bar and select **Blueprint**.
3. Connect your GitHub account and select your `vortex-ai-chatbot` repository.
4. Render will automatically detect `render.yaml` and configure:
   - **Environment:** `Python 3.11.9`
   - **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Health Check:** `/health`
5. In the prompt for environment variables, fill in your `GEMINI_API_KEY`.
6. Click **Apply**. Render will build and deploy your app with a public `https://<service-name>.onrender.com` URL.

---

### Method B: Manual Web Service Setup

If you prefer manual configuration:

1. In Render Dashboard, click **New +** → **Web Service**.
2. Connect your GitHub repository.
3. Configure the following settings:
   - **Name:** `vortex-ai-chatbot`
   - **Language:** `Python 3`
   - **Branch:** `main` (or `master`)
   - **Region:** Choose closest to your users (e.g., Oregon, Frankfurt, Singapore)
   - **Build Command:**
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     gunicorn app:app
     ```
   - **Instance Type:** `Free`
4. Expand **Advanced** and set:
   - **Health Check Path:** `/health`
   - **Auto-Deploy:** `Yes`
5. In the **Environment Variables** section, add:
   | Key | Value | Notes |
   | :--- | :--- | :--- |
   | `PYTHON_VERSION` | `3.11.9` | Ensures exact Python runtime |
   | `GEMINI_API_KEY` | `your_actual_gemini_api_key` | Secret API key from Google AI Studio |
   | `GEMINI_MODEL` | `gemini-3.5-flash` | Default Gemini model |
   | `LLM_PROVIDER` | `auto` | Auto routes between FAQ & Gemini |
   | `SIMILARITY_THRESHOLD` | `0.35` | 0.10 to 0.80 range |
   | `FLASK_DEBUG` | `false` | Production mode |
6. Click **Create Web Service**.
7. Wait 1–2 minutes for the build to finish. Once live, open your Render URL!

---

## 🩺 Health Check & Monitoring

Once deployed, you can verify your service health at:
```http
GET https://<your-render-app>.onrender.com/health
```
Response:
```json
{
  "status": "healthy",
  "service": "CodeAlpha Hybrid AI FAQ Chatbot",
  "timestamp": 1792888896.81,
  "uptime": 3816.58
}
```

---

## 🔒 Security & Best Practices

- **Zero API Key Leakage:** API keys are read strictly from runtime environment variables on the backend. Frontend code never accesses or exposes server secrets.
- **Git Protection:** `.gitignore` actively prevents `.env`, local caches, and logs from being committed.
- **AST Safe Math Execution:** Dynamic math evaluations use abstract syntax tree inspection to eliminate `eval()` injection vulnerabilities.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
