"""
nlp_engine.py
=============================================================================
Unified Hybrid AI NLP Coordinator Engine
-----------------------------------------------------------------------------
Coordinates:
1. Tool 1: Live Safe Math & Expression Solver (AST Safe Eval)
2. Tool 2: Real-time Date, Time & Utility Intent Engine
3. Tool 3 & 4: FAQ Engine (Alias + Dual-Channel TF-IDF Cosine Similarity)
4. Tool 5: LLM Engine (Google Gemini / OpenAI with Multi-Turn Conversation History)
5. Tool 6: Built-in Offline Knowledge Encyclopedia Fallback
=============================================================================
"""

import ast
import datetime
import logging
import math
import operator
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
from faq_engine import FAQEngine, preprocess_text
from llm_engine import LLMEngine

logger = logging.getLogger("HybridNLPEngine")

# Safe Math AST Evaluator
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos
}

SAFE_FUNCTIONS = {
    "sqrt": math.isqrt,
    "abs": abs,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp
}


def safe_eval_ast(node):
    """Safely evaluates an AST math expression node without using eval()."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        left = safe_eval_ast(node.left)
        right = safe_eval_ast(node.right)
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported math operator: {op_type}")
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Division by zero")
        return SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval_ast(node.operand)
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type}")
        return SAFE_OPERATORS[op_type](operand)
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
            args = [safe_eval_ast(arg) for arg in node.args]
            return SAFE_FUNCTIONS[node.func.id](*args)
        raise ValueError("Unsupported function call")
    raise ValueError(f"Unsupported AST node: {type(node)}")


class HybridNLPEngine:
    """
    Main Hybrid NLP Engine uniting deterministic FAQ matching with Generative AI.
    """

    def __init__(self, faqs_path: Optional[str] = None, default_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD):
        self.default_threshold = default_threshold
        self.faqs_path = Path(faqs_path) if faqs_path else config.FAQS_FILE

        # Child engines
        self.faq_engine = FAQEngine(faqs_path=self.faqs_path, default_threshold=default_threshold)
        self.llm_engine = LLMEngine()

    @property
    def faqs(self) -> List[Dict[str, Any]]:
        """Backward compatibility property returning loaded FAQs list."""
        return self.faq_engine.faqs

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Backward compatibility static method for text preprocessing."""
        return preprocess_text(text)

    def load_faqs_and_fit(self) -> int:
        """Backward compatibility method to reload FAQs and fit vectorizers."""
        return self.faq_engine.load_and_fit()

    def match_alias_exact(self, query: str) -> Optional[Tuple[Dict[str, Any], float]]:
        """Backward compatibility method for direct alias matching."""
        return self.faq_engine.match_alias_exact(query)

    def match_local_faq(self, query: str, category_filter: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], float, List[Dict[str, Any]]]:
        """Backward compatibility method for matching local FAQs."""
        return self.faq_engine.match_query(query, category_filter=category_filter)

    def generate_ai_fallback(self, query: str, top_candidates: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, str]:
        """Backward compatibility method for AI generation/fallback."""
        return self.llm_engine.generate_response(query=query, top_candidates=top_candidates)

    # ==========================================================================
    # TOOL 1: LIVE MATH & EXPRESSION SOLVER (SAFE AST)
    # ==========================================================================
    def solve_math_query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Detects and calculates mathematical queries in natural language or symbols.
        Supports:
          - "what is 2 times 4" -> 8
          - "150 + 350 / 5" -> 220
          - "square root of 144" -> 12
          - "25% of 800" -> 200
        """
        q = query.lower().strip()
        math_keywords = [
            "times", "multiplied by", "divided by", "plus", "minus", "power of",
            "square root", "sqrt", "% of", "percent of", "+", "-", "*", "/", "%", "^", "calculate", "solve"
        ]
        if not any(k in q for k in math_keywords) and not re.search(r"\d+\s*[\+\-\*\/\^\%]\s*\d+", q):
            return None

        cleaned = re.sub(r"^(?:what\s+is|what\'s|calculate|compute|solve|evaluate|please\s+calculate|tell\s+me)\s+", "", q).strip()
        cleaned = cleaned.rstrip("?=! .")

        # 1. Square root pattern: "square root of 144"
        sqrt_match = re.search(r"(?:square\s*root\s*of|sqrt\s*of|sqrt)\s*([0-9\.]+)", cleaned)
        if sqrt_match:
            try:
                num = float(sqrt_match.group(1))
                res = math.sqrt(num)
                res_fmt = int(res) if res.is_integer() else round(res, 4)
                return {
                    "response": (
                        f"### 🧮 Math & Logic Engine\n\n"
                        f"$$\\sqrt{{{int(num) if num.is_integer() else num}}} = {res_fmt}$$\n\n"
                        f"**Result:** `{res_fmt}`"
                    ),
                    "source": "🧮 Math & Logic Engine",
                    "confidence": 100.0,
                    "matched_faq": None
                }
            except Exception:
                pass

        # 2. Percentage pattern: "25% of 800"
        pct_match = re.search(r"([0-9\.]+)\s*(?:%|percent)\s*(?:of)\s*([0-9\.]+)", cleaned)
        if pct_match:
            try:
                pct = float(pct_match.group(1))
                base = float(pct_match.group(2))
                res = (pct / 100.0) * base
                res_fmt = int(res) if res.is_integer() else round(res, 4)
                return {
                    "response": (
                        f"### 🧮 Math & Logic Engine\n\n"
                        f"$$\\frac{{{pct}}}{{100}} \\times {base} = {res_fmt}$$\n\n"
                        f"**Result:** `{res_fmt}`"
                    ),
                    "source": "🧮 Math & Logic Engine",
                    "confidence": 100.0,
                    "matched_faq": None
                }
            except Exception:
                pass

        # 3. Natural language operator translation
        expr = cleaned
        expr = re.sub(r"\bmultiplied\s*by\b", "*", expr)
        expr = re.sub(r"\btimes\b", "*", expr)
        expr = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", expr)
        expr = re.sub(r"\bdivided\s*by\b", "/", expr)
        expr = re.sub(r"\bover\b", "/", expr)
        expr = re.sub(r"\bplus\b", "+", expr)
        expr = re.sub(r"\badd\b", "+", expr)
        expr = re.sub(r"\bminus\b", "-", expr)
        expr = re.sub(r"\bsubtract\b", "-", expr)
        expr = re.sub(r"\bto\s*the\s*power\s*of\b", "**", expr)
        expr = re.sub(r"\braised\s*to\b", "**", expr)
        expr = re.sub(r"\bpower\s*of\b", "**", expr)
        expr = re.sub(r"\^", "**", expr)
        expr = re.sub(r"\bmod\b|\bmodulo\b", "%", expr)

        sanitized = re.sub(r"[^0-9\+\-\*\/\%\(\)\.\s]", "", expr).strip()
        if not re.search(r"\d", sanitized) or not re.search(r"[\+\-\*\/\%]", sanitized):
            return None

        try:
            parsed_tree = ast.parse(sanitized, mode="eval")
            result_val = safe_eval_ast(parsed_tree.body)

            if isinstance(result_val, float):
                formatted_res = int(result_val) if result_val.is_integer() else round(result_val, 6)
            else:
                formatted_res = result_val

            display_expr = sanitized.replace("**", "^").replace("*", " × ").replace("/", " ÷ ")

            return {
                "response": (
                    f"### 🧮 Math & Logic Engine\n\n"
                    f"$$\\text{{{display_expr}}} = \\mathbf{{{formatted_res}}}$$\n\n"
                    f"**Calculated Result:** `{formatted_res}`"
                ),
                "source": "🧮 Math & Logic Engine",
                "confidence": 100.0,
                "matched_faq": None
            }
        except Exception:
            return None

    # ==========================================================================
    # TOOL 2: REAL-TIME DATE, TIME & UTILITY ENGINE
    # ==========================================================================
    def check_utility_intents(self, query: str) -> Optional[Dict[str, Any]]:
        """Handles current time, today's date, coin flips, dice rolls, etc."""
        q = query.lower().strip().rstrip("?!. ")

        # 1. Current Time
        if re.search(r"\b(?:what(?:\s*is)?\s*(?:the\s*)?time|current\s*time|time\s*now|what\s*time\s*is\s*it)\b", q):
            now = datetime.datetime.now()
            time_str = now.strftime("%I:%M:%S %p")
            date_str = now.strftime("%A, %B %d, %Y")
            return {
                "response": (
                    f"### ⏱️ Real-Time Clock\n\n"
                    f"- **Current Time:** `{time_str}`\n"
                    f"- **Date:** {date_str}\n"
                    f"- **System Clock:** Local System Time Active"
                ),
                "source": "⏱️ Real-time Utility Engine",
                "confidence": 100.0,
                "matched_faq": None
            }

        # 2. Today's Date
        if re.search(r"\b(?:what(?:\s*is)?\s*(?:today\'?s?\s*)?date|current\s*date|today\'?s?\s*date|what\s*day\s*is\s*(?:it|today)|day\s*of\s*(?:the\s*)?week)\b", q):
            now = datetime.datetime.now()
            date_str = now.strftime("%A, %B %d, %Y")
            day_name = now.strftime("%A")
            return {
                "response": (
                    f"### 📅 Calendar Service\n\n"
                    f"- **Today's Date:** **{date_str}**\n"
                    f"- **Day of the Week:** **{day_name}**"
                ),
                "source": "⏱️ Real-time Utility Engine",
                "confidence": 100.0,
                "matched_faq": None
            }

        # 3. Flip a Coin
        if re.search(r"\b(?:flip\s*a\s*coin|coin\s*toss|heads\s*or\s*tails)\b", q):
            outcome = random.choice(["Heads 🪙", "Tails 🪙"])
            return {
                "response": (
                    f"### 🪙 Coin Toss Result\n\n"
                    f"The coin landed on: **{outcome}**!"
                ),
                "source": "⏱️ Real-time Utility Engine",
                "confidence": 100.0,
                "matched_faq": None
            }

        # 4. Roll a Die
        if re.search(r"\b(?:roll\s*(?:a\s*)?(?:die|dice))\b", q):
            roll = random.randint(1, 6)
            dice_icons = {1: "⚀", 2: "⚁", 3: "⚂", 4: "⚃", 5: "⚄", 6: "⚅"}
            return {
                "response": (
                    f"### 🎲 Dice Roll Result\n\n"
                    f"You rolled a **{roll}** {dice_icons.get(roll, '')}!"
                ),
                "source": "⏱️ Real-time Utility Engine",
                "confidence": 100.0,
                "matched_faq": None
            }

        return None

    # ==========================================================================
    # MAIN QUERY PROCESSOR (ROUTING LOGIC)
    # ==========================================================================
    def process_query(
        self,
        query: str,
        mode: str = "auto",
        threshold: Optional[float] = None,
        category_filter: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Executes hybrid routing across Auto, FAQ Only, and AI Assistant modes.
        Modes:
          - 'auto' / 'hybrid': Matches FAQ; routes to LLM if below threshold.
          - 'faq' / 'faq_only': Strictly returns verified FAQs; NEVER calls LLM.
          - 'ai' / 'ai_assistant' / 'genai': Routes directly to LLM with conversation history.
        """
        start_time = time.perf_counter()
        t = threshold if threshold is not None else self.default_threshold
        t = max(0.01, min(0.99, float(t)))

        # Normalize mode names
        mode_lower = mode.lower().strip()
        if mode_lower in ("auto", "hybrid"):
            resolved_mode = "auto"
        elif mode_lower in ("faq", "faq_only", "faq only", "strict_faq"):
            resolved_mode = "faq"
        elif mode_lower in ("ai", "ai_assistant", "ai assistant", "genai"):
            resolved_mode = "ai"
        else:
            resolved_mode = "auto"

        query_clean = query.strip()
        if not query_clean:
            return {
                "response": "Please type a question or select one of the suggested prompts!",
                "source": "System",
                "confidence": 0.0,
                "matched_faq": None,
                "latency_ms": 0.0,
                "mode": resolved_mode,
                "threshold": t,
                "candidates": []
            }

        # 1. TOOL 1: Check Live Math Solver
        math_result = self.solve_math_query(query_clean)
        if math_result is not None:
            latency = (time.perf_counter() - start_time) * 1000
            math_result["latency_ms"] = round(latency, 2)
            math_result["mode"] = resolved_mode
            math_result["threshold"] = t
            math_result["candidates"] = []
            return math_result

        # 2. TOOL 2: Check Real-Time Date/Time & Utilities
        util_result = self.check_utility_intents(query_clean)
        if util_result is not None:
            latency = (time.perf_counter() - start_time) * 1000
            util_result["latency_ms"] = round(latency, 2)
            util_result["mode"] = resolved_mode
            util_result["threshold"] = t
            util_result["candidates"] = []
            return util_result

        # ======================================================================
        # MODE: AI Assistant (Direct LLM with conversation history)
        # ======================================================================
        if resolved_mode == "ai":
            answer, source = self.llm_engine.generate_response(query=query_clean, history=history)
            latency = (time.perf_counter() - start_time) * 1000
            return {
                "response": answer,
                "source": source,
                "confidence": 95.0 if "Gemini" in source or "OpenAI" in source else 85.0,
                "matched_faq": None,
                "latency_ms": round(latency, 2),
                "mode": resolved_mode,
                "threshold": t,
                "candidates": []
            }

        # ======================================================================
        # FAQ MATCHING (For Auto Mode and FAQ Only Mode)
        # ======================================================================
        best_faq, best_score, candidates = self.faq_engine.match_query(
            query=query_clean,
            category_filter=category_filter
        )

        # ======================================================================
        # MODE: FAQ Only (Strictly NEVER calls LLM)
        # ======================================================================
        if resolved_mode == "faq":
            latency = (time.perf_counter() - start_time) * 1000
            if best_faq and best_score >= t:
                return {
                    "response": best_faq["answer"],
                    "source": "📚 Local FAQ Match",
                    "confidence": round(best_score * 100, 1),
                    "matched_faq": {
                        "id": best_faq.get("id"),
                        "question": best_faq.get("question"),
                        "category": best_faq.get("category"),
                        "tags": best_faq.get("tags", [])
                    },
                    "latency_ms": round(latency, 2),
                    "mode": resolved_mode,
                    "threshold": t,
                    "candidates": candidates
                }
            else:
                top_hint = (
                    f" Closest FAQ candidate was *'{candidates[0]['question']}'* "
                    f"with {round(candidates[0]['similarity_score'] * 100, 1)}% similarity."
                ) if candidates else ""

                return {
                    "response": (
                        f"⚠️ **No verified FAQ match found above threshold `{round(t * 100, 1)}%`** "
                        f"(Best match: `{round(best_score * 100, 1)}%`).\n\n"
                        f"{top_hint}\n\n"
                        f"*(Strict FAQ mode is active — external LLMs are not called. "
                        f"Switch to **Auto Mode** or **AI Assistant Mode** to query general topics).* "
                    ),
                    "source": "Strict FAQ Matcher",
                    "confidence": round(best_score * 100, 1),
                    "matched_faq": None,
                    "latency_ms": round(latency, 2),
                    "mode": resolved_mode,
                    "threshold": t,
                    "candidates": candidates
                }

        # ======================================================================
        # MODE: Auto (Default: FAQ if >= threshold; else General LLM)
        # ======================================================================
        if best_faq and best_score >= t:
            latency = (time.perf_counter() - start_time) * 1000
            return {
                "response": best_faq["answer"],
                "source": "📚 Local FAQ Match",
                "confidence": round(best_score * 100, 1),
                "matched_faq": {
                    "id": best_faq.get("id"),
                    "question": best_faq.get("question"),
                    "category": best_faq.get("category"),
                    "tags": best_faq.get("tags", [])
                },
                "latency_ms": round(latency, 2),
                "mode": resolved_mode,
                "threshold": t,
                "candidates": candidates
            }

        # Query below threshold -> Route to General AI Engine
        answer, source = self.llm_engine.generate_response(
            query=query_clean,
            history=history,
            top_candidates=candidates
        )
        latency = (time.perf_counter() - start_time) * 1000

        return {
            "response": answer,
            "source": source,
            "confidence": round(best_score * 100, 1) if best_score > 0 else 0.0,
            "matched_faq": {
                "id": best_faq.get("id") if best_faq else None,
                "question": best_faq.get("question") if best_faq else None,
                "category": best_faq.get("category") if best_faq else None
            } if best_faq else None,
            "latency_ms": round(latency, 2),
            "mode": resolved_mode,
            "threshold": t,
            "candidates": candidates
        }

    # ==========================================================================
    # DYNAMIC FAQ MANAGEMENT
    # ==========================================================================
    def add_custom_faq(self, question: str, answer: str, category: str = "custom", tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Dynamically adds a custom FAQ and re-fits TF-IDF live."""
        return self.faq_engine.add_faq(question=question, answer=answer, category=category, tags=tags)

    def edit_custom_faq(self, faq_id: str, question: str, answer: str, category: Optional[str] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Dynamically edits an existing FAQ and re-fits TF-IDF live."""
        return self.faq_engine.edit_faq(faq_id=faq_id, question=question, answer=answer, category=category, tags=tags)

    def delete_custom_faq(self, faq_id: str) -> Dict[str, Any]:
        """Dynamically deletes an FAQ and re-fits TF-IDF live."""
        return self.faq_engine.delete_faq(faq_id=faq_id)

    def get_stats(self) -> Dict[str, Any]:
        """Returns unified system statistics."""
        faq_stats = self.faq_engine.get_stats()
        llm_stats = self.llm_engine.get_stats()

        return {
            "total_faqs": faq_stats["total_faqs"],
            "categories": faq_stats["categories"],
            "vocabulary_size": faq_stats["vocabulary_size"],
            "char_features": faq_stats["char_features"],
            "default_threshold": self.default_threshold,
            "llm_provider": llm_stats["active_provider"],
            "gemini_active": llm_stats["gemini_active"],
            "gemini_model": llm_stats["gemini_model"],
            "openai_active": llm_stats["openai_active"],
            "openai_model": llm_stats["openai_model"],
            "model_name": (
                f"Google Gemini ({llm_stats['gemini_model']})" if llm_stats["gemini_active"]
                else (f"OpenAI ({llm_stats['openai_model']})" if llm_stats["openai_active"]
                      else "Smart Offline Knowledge Engine")
            )
        }
