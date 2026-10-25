"""
llm_engine.py
=============================================================================
General AI LLM Integration Engine (Google Gemini + OpenAI)
-----------------------------------------------------------------------------
Features:
- Configurable multi-provider architecture: Google Gemini (google-genai) and OpenAI
- Multi-turn conversation history management for follow-up context
- Graceful handling of timeouts, rate limits, network outages, and missing keys
- Zero-key smart offline knowledge resolver fallback (never crashes or shows placeholders)
- Safe handling of API credentials (never logged or sent to client)
=============================================================================
"""

import json
import logging
import os
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

import config

logger = logging.getLogger("LLMEngine")

SYSTEM_INSTRUCTION = (
    "You are Vortex, an articulate, highly capable, and encouraging AI Assistant. "
    "You provide clear, accurate, and comprehensive explanations with well-structured Markdown, "
    "bullet points, formulas, and runnable code examples where appropriate. "
    "Help users with programming (Python, C++, Java, JavaScript, etc.), debugging, data science, "
    "machine learning, system design, writing, summarization, and general educational questions. "
    "Maintain context across the recent conversation history. "
    "Never pretend to know an answer when uncertain; state any limitations clearly and objectively."
)


class LLMEngine:
    """
    Coordinates Generative AI query execution across Google Gemini, OpenAI,
    or the Zero-Key Smart Offline Knowledge Resolver.
    """

    def __init__(self):
        self.provider = config.LLM_PROVIDER
        self.gemini_key = config.GEMINI_API_KEY
        self.gemini_model = config.GEMINI_MODEL
        self.openai_key = config.OPENAI_API_KEY
        self.openai_model = config.OPENAI_MODEL
        self.max_history = config.MAX_HISTORY_TURNS

        self.gemini_client = None
        self.openai_client = None

        self._init_clients()

    def _init_clients(self):
        """Initializes client SDKs if keys are present."""
        # Initialize Google GenAI Client
        if self.gemini_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.gemini_key)
                logger.info("Initialized Google Gemini client SDK successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize google-genai SDK: {e}. Will attempt REST if needed.")
                self.gemini_client = None

        # Initialize OpenAI Client (if package installed)
        if self.openai_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=self.openai_key)
                logger.info("Initialized OpenAI official SDK client successfully.")
            except ImportError:
                logger.info("OpenAI package not installed. Will use native HTTPS REST client for OpenAI.")
                self.openai_client = None
            except Exception as e:
                logger.warning(f"OpenAI client init warning: {e}. Using native HTTPS REST fallback.")
                self.openai_client = None

    def get_active_provider(self) -> str:
        """Determines which provider is currently active based on config and available keys."""
        if self.provider == "gemini" and self.gemini_key:
            return "gemini"
        if self.provider == "openai" and self.openai_key:
            return "openai"

        # Auto detection
        if self.gemini_key:
            return "gemini"
        if self.openai_key:
            return "openai"

        return "offline"

    def _normalize_history(self, raw_history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        """Converts heterogeneous client history formats into standard [{role: 'user'|'assistant', content: '...'}] format."""
        if not raw_history or not isinstance(raw_history, list):
            return []

        normalized = []
        for turn in raw_history[-self.max_history:]:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role") or ("user" if turn.get("sender") == "user" else "assistant")
            content = turn.get("content") or turn.get("text") or ""
            content = str(content).strip()
            if content and role in ("user", "assistant"):
                normalized.append({"role": role, "content": content})

        return normalized

    def generate_response(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
        top_candidates: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, str]:
        """
        Routes the prompt to the configured LLM provider or offline fallback.
        Returns: (response_text, source_display_label)
        """
        active = self.get_active_provider()
        history_normalized = self._normalize_history(history)

        if active == "gemini":
            ans, src = self._call_gemini(query, history_normalized)
            if ans:
                return ans, src

        elif active == "openai":
            ans, src = self._call_openai(query, history_normalized)
            if ans:
                return ans, src

        # Fallback to Smart Offline Knowledge Engine
        return self._offline_resolver(query, top_candidates), "⚡ Smart Knowledge Engine (Offline Mode)"

    def _call_gemini(self, query: str, history: List[Dict[str, str]]) -> Tuple[Optional[str], str]:
        """Invokes Google Gemini with multi-turn conversation context."""
        try:
            if self.gemini_client:
                # Format conversation contents for google-genai
                contents_payload = []
                for turn in history:
                    contents_payload.append({
                        "role": "user" if turn["role"] == "user" else "model",
                        "parts": [{"text": turn["content"]}]
                    })
                contents_payload.append({
                    "role": "user",
                    "parts": [{"text": query}]
                })

                models_to_try = [self.gemini_model, "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest"]
                unique_models = []
                for m in models_to_try:
                    if m and m not in unique_models:
                        unique_models.append(m)

                last_error = None
                for candidate_model in unique_models:
                    try:
                        response = self.gemini_client.models.generate_content(
                            model=candidate_model,
                            contents=contents_payload,
                            config={
                                "system_instruction": SYSTEM_INSTRUCTION,
                                "temperature": 0.4
                            }
                        )
                        if response and response.text:
                            return response.text.strip(), f"✨ Gemini AI ({candidate_model})"
                    except Exception as model_err:
                        last_error = model_err
                        logger.warning(f"Gemini model {candidate_model} notice: {model_err}. Trying alternate model...")
                        continue

                if last_error:
                    raise last_error

        except Exception as e:
            err_msg = str(e).lower()
            logger.error(f"Gemini API invocation error: {e}")
            if "quota" in err_msg or "rate limit" in err_msg or "resource_exhausted" in err_msg:
                return (
                    "⚠️ **Google Gemini Rate Limit Reached:** The API quota for the Gemini model has been temporarily exceeded.\n\n"
                    "Please wait a moment before sending another prompt, or check your API key quota at [Google AI Studio](https://aistudio.google.com/).",
                    "⚠️ Gemini Rate Limit"
                )
            if "api key" in err_msg or "unauthenticated" in err_msg:
                return (
                    "⚠️ **Gemini Authentication Error:** The configured `GEMINI_API_KEY` appears invalid or expired.\n\n"
                    "Please verify your API key in `.env`.",
                    "⚠️ Gemini Auth Error"
                )

        return None, ""

    def _call_openai(self, query: str, history: List[Dict[str, str]]) -> Tuple[Optional[str], str]:
        """Invokes OpenAI with multi-turn conversation context."""
        messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": query})

        # 1. Use SDK if available
        if self.openai_client:
            try:
                completion = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=0.4,
                    timeout=15.0
                )
                if completion.choices and completion.choices[0].message.content:
                    return completion.choices[0].message.content.strip(), f"✨ OpenAI ({self.openai_model})"
            except Exception as e:
                logger.error(f"OpenAI SDK error: {e}")
                err_str = str(e).lower()
                if "rate limit" in err_str or "quota" in err_str:
                    return "⚠️ **OpenAI Rate Limit:** Your OpenAI quota has been exceeded or rate-limited.", "⚠️ OpenAI Rate Limit"
                if "authentication" in err_str or "api key" in err_str:
                    return "⚠️ **OpenAI Auth Error:** The configured `OPENAI_API_KEY` is invalid.", "⚠️ OpenAI Auth Error"

        # 2. Native HTTPS REST fallback for OpenAI
        try:
            req_data = json.dumps({
                "model": self.openai_model,
                "messages": messages,
                "temperature": 0.4
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.openai_key}"
                },
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=12) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                text = res_body["choices"][0]["message"]["content"]
                return text.strip(), f"✨ OpenAI ({self.openai_model})"

        except urllib.error.HTTPError as e:
            logger.error(f"OpenAI REST HTTP error: {e.code} - {e.reason}")
            if e.code == 429:
                return "⚠️ **OpenAI Rate Limit (429):** Quota exceeded. Please verify billing on platform.openai.com.", "⚠️ OpenAI Rate Limit"
            if e.code in (401, 403):
                return "⚠️ **OpenAI Auth Error (401):** Invalid API Key in `.env`.", "⚠️ OpenAI Auth Error"
        except Exception as e:
            logger.error(f"OpenAI REST request failed: {e}")

        return None, ""

    def _offline_resolver(self, query: str, top_candidates: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Smart Zero-Key Offline Encyclopedia:
        Answers common CS, AI, programming, and general questions directly,
        or serves the closest matching verified FAQ candidate without returning generic placeholders.
        """
        q = query.lower().strip()

        # If we have a reasonably close candidate (>= 0.18), present its verified answer
        if top_candidates and len(top_candidates) > 0:
            best_cand = top_candidates[0]
            if best_cand.get("similarity_score", 0) >= 0.18:
                return (
                    f"### 📚 Knowledge Base Match\n\n"
                    f"*Closely matched topic: **{best_cand.get('question')}***\n\n"
                    f"{best_cand.get('answer')}"
                )

        # Built-in CS & Programming Topics
        if re.search(r"\b(recursion|recursive)\b", q):
            return (
                "### 🔁 What is Recursion?\n\n"
                "**Recursion** is a programming paradigm where a function calls itself to solve smaller subproblems of the same type.\n\n"
                "**Two Core Elements:**\n"
                "1. **Base Case:** The terminating condition that halts further recursive calls.\n"
                "2. **Recursive Step:** Progressively reduces the problem towards the base case.\n\n"
                "```python\ndef factorial(n: int) -> int:\n    if n <= 1:\n        return 1  # Base case\n    return n * factorial(n - 1)  # Recursive call\n```"
            )

        if re.search(r"\b(binary\s*search)\b", q):
            return (
                "### 🔍 Binary Search Algorithm\n\n"
                "**Binary Search** is an optimal $O(\\log N)$ search algorithm for **sorted collections** that halves the search space at each iteration.\n\n"
                "```python\ndef binary_search(arr: list, target: int) -> int:\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = low + (high - low) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n```"
            )

        if re.search(r"\b(what\s*is\s*api|define\s*api|explain\s*api)\b", q):
            return (
                "### 🔌 What is an API?\n\n"
                "An **API (Application Programming Interface)** defines rules and contracts enabling distinct software applications to communicate.\n\n"
                "- **Client:** Requests a resource (e.g., frontend browser making a `GET` request).\n"
                "- **Server:** Processes requests, interacts with databases, and returns responses (commonly JSON).\n"
                "- **REST Principles:** Uses HTTP verbs (`GET`, `POST`, `PUT`, `DELETE`) with stateless interactions."
            )

        if re.search(r"\b(cloud\s*computing|what\s*is\s*cloud)\b", q):
            return (
                "### ☁️ Cloud Computing\n\n"
                "**Cloud Computing** is the on-demand delivery of computing services (servers, storage, networking, AI) over the internet with pay-as-you-go pricing.\n\n"
                "- **IaaS:** AWS EC2, Google Compute Engine\n"
                "- **PaaS:** Render, AWS Elastic Beanstalk, Heroku\n"
                "- **SaaS:** Google Workspace, Slack, Microsoft 365"
            )

        # Detect random gibberish
        vowels = len(re.findall(r"[aeiouy]", q))
        if len(q) > 6 and (vowels == 0 or len(set(q)) <= 3):
            return (
                "⚠️ **Input Not Recognized:** The query appears to consist of random characters or keystrokes.\n\n"
                "Please ask a question such as:\n"
                "- *\"What is Artificial Intelligence?\"*\n"
                "- *\"What is the mandatory attendance policy?\"*\n"
                "- *\"Write a Python function for Fibonacci numbers\"*"
            )

        # Helpful suggestions
        hints = ""
        if top_candidates:
            hint_items = [f"- **{c.get('question')}**" for c in top_candidates[:3]]
            hints = "\n\n**Related verified topics you can explore:**\n" + "\n".join(hint_items)

        return (
            f"### 💡 AI Assistant Knowledge Hub\n\n"
            f"I analyzed your question regarding: *\"{query}\"*.\n\n"
            f"This query is currently outside our pre-indexed FAQ dataset. "
            f"You can add it directly to our verified knowledge base using the **`+ Add Custom FAQ`** button in the header!\n"
            f"{hints}\n\n"
            f"*(Tip: To enable open-domain web-scale generative responses for any arbitrary topic, "
            f"set `GEMINI_API_KEY` or `OPENAI_API_KEY` in your `.env` file).* "
        )

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on active LLM provider and available keys."""
        return {
            "configured_provider": self.provider,
            "active_provider": self.get_active_provider(),
            "gemini_active": bool(self.gemini_key),
            "gemini_model": self.gemini_model,
            "openai_active": bool(self.openai_key),
            "openai_model": self.openai_model
        }
