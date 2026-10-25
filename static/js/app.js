/**
 * app.js
 * =============================================================================
 * Interactive Client Logic for CodeAlpha Hybrid AI FAQ Chatbot
 * =============================================================================
 */

document.addEventListener("DOMContentLoaded", () => {
  // --- APPLICATION STATE ---
  const state = {
    mode: "auto",             // 'auto' | 'faq' | 'ai'
    threshold: 0.35,          // 0.10 to 0.80
    category: "all",          // 'all' | 'university_campus' | etc.
    isGenerating: false,
    history: [],              // Stored message turns of active conversation
    currentSessionId: null,   // Active chat session ID
    sessions: []              // All stored chat sessions
  };

  // --- DOM SELECTORS ---
  const messagesContainer = document.getElementById("messagesContainer");
  const welcomeHero = document.getElementById("welcomeHero");
  const typingIndicator = document.getElementById("typingIndicator");
  const typingLabel = document.getElementById("typingLabel");
  const chatForm = document.getElementById("chatForm");
  const messageInput = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");

  // Sidebar Controls & Chat History
  const sidebar = document.getElementById("sidebar");
  const sidebarBackdrop = document.getElementById("sidebarBackdrop");
  const mobileMenuBtn = document.getElementById("mobileMenuBtn");
  const newChatBtn = document.getElementById("newChatBtn");
  const historySearchInput = document.getElementById("historySearchInput");
  const clearSearchHistoryBtn = document.getElementById("clearSearchHistoryBtn");
  const clearAllHistoryBtn = document.getElementById("clearAllHistoryBtn");
  const historySessionsList = document.getElementById("historySessionsList");
  const historyEmptyState = document.getElementById("historyEmptyState");
  const historyEmptyText = document.getElementById("historyEmptyText");

  // Input Tray Mode Switcher
  const engineModePills = document.getElementById("engineModePills");
  const modeDescription = document.getElementById("modeDescription");
  const thresholdSlider = document.getElementById("thresholdSlider");
  const thresholdValue = document.getElementById("thresholdValue");

  // Header Controls
  const activeModeBadge = document.getElementById("activeModeBadge");
  const headerLatencyBadge = document.getElementById("headerLatencyBadge");
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const clearChatBtn = document.getElementById("clearChatBtn");
  const exportDropdownBtn = document.getElementById("exportDropdownBtn");
  const exportDropdownMenu = document.getElementById("exportDropdownMenu");
  const exportTxtBtn = document.getElementById("exportTxtBtn");
  const exportJsonBtn = document.getElementById("exportJsonBtn");

  // Stats
  const statTotalFaqs = document.getElementById("statTotalFaqs");
  const statLlmModel = document.getElementById("statLlmModel");

  // Manage / Edit / Add FAQ Modal
  const openManageFaqBtn = document.getElementById("openManageFaqBtn");
  const faqManagerModal = document.getElementById("faqManagerModal");
  const closeFaqManagerModalBtn = document.getElementById("closeFaqManagerModalBtn");
  const cancelFaqBtn = document.getElementById("cancelFaqBtn");
  const tabAddFaqBtn = document.getElementById("tabAddFaqBtn");
  const tabBrowseFaqBtn = document.getElementById("tabBrowseFaqBtn");
  const tabContentAdd = document.getElementById("tabContentAdd");
  const tabContentBrowse = document.getElementById("tabContentBrowse");
  const addFaqForm = document.getElementById("addFaqForm");
  const editFaqId = document.getElementById("editFaqId");
  const saveBtnText = document.getElementById("saveBtnText");
  const formNoticeText = document.getElementById("formNoticeText");
  const modalMainTitle = document.getElementById("modalMainTitle");
  const faqSearchInput = document.getElementById("faqSearchInput");
  const faqCategoryFilterSelect = document.getElementById("faqCategoryFilterSelect");
  const faqsListScrollable = document.getElementById("faqsListScrollable");

  // Toast Container
  const toastContainer = document.getElementById("toastContainer");

  // Quick Chips & UX Selectors
  const clearInputBtn = document.getElementById("clearInputBtn");
  const scrollBottomBtn = document.getElementById("scrollBottomBtn");

  // --- THEME INITIALIZATION ---
  initTheme();

  function initTheme() {
    const savedTheme = localStorage.getItem("codealpha_theme") || "theme-light";
    document.body.className = savedTheme;
    updateThemeIcons(savedTheme);

    if (themeToggleBtn) {
      themeToggleBtn.addEventListener("click", () => {
        const isDark = document.body.classList.contains("theme-dark");
        const nextTheme = isDark ? "theme-light" : "theme-dark";
        document.body.className = nextTheme;
        localStorage.setItem("codealpha_theme", nextTheme);
        updateThemeIcons(nextTheme);
        showToast(`Theme switched to ${isDark ? "Light" : "Dark"} Mode`, "info");
      });
    }
  }

  function updateThemeIcons(currentTheme) {
    if (!themeToggleBtn) return;
    const sunIcon = themeToggleBtn.querySelector(".sun-icon");
    const moonIcon = themeToggleBtn.querySelector(".moon-icon");
    if (currentTheme === "theme-light") {
      sunIcon.classList.add("hidden");
      moonIcon.classList.remove("hidden");
    } else {
      sunIcon.classList.remove("hidden");
      moonIcon.classList.add("hidden");
    }
  }

  // --- MARKED.JS CONFIGURATION ---
  if (typeof marked !== "undefined") {
    marked.setOptions({
      breaks: true,
      gfm: true,
      headerIds: false,
      mangle: false
    });
  }

  // --- INITIALIZE & FETCH STATS ---
  fetchEngineStats();

  // --- AUTO RESIZE TEXTAREA & CLEAR BUTTON TOGGLE ---
  messageInput.addEventListener("input", () => {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 160) + "px";
    const hasText = !!messageInput.value.trim();
    sendBtn.disabled = !hasText || state.isGenerating;
    if (clearInputBtn) {
      clearInputBtn.classList.toggle("hidden", !messageInput.value);
    }
  });

  if (clearInputBtn) {
    clearInputBtn.addEventListener("click", () => {
      messageInput.value = "";
      messageInput.style.height = "auto";
      sendBtn.disabled = true;
      clearInputBtn.classList.add("hidden");
      messageInput.focus();
    });
  }


  // Scroll to bottom button
  if (messagesContainer && scrollBottomBtn) {
    messagesContainer.addEventListener("scroll", () => {
      const isUp = messagesContainer.scrollHeight - messagesContainer.scrollTop - messagesContainer.clientHeight > 140;
      scrollBottomBtn.classList.toggle("hidden", !isUp);
    });

    scrollBottomBtn.addEventListener("click", () => {
      scrollToBottom();
    });
  }

  // Enter to send, Shift+Enter for newline
  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!sendBtn.disabled) {
        chatForm.dispatchEvent(new Event("submit"));
      }
    }
  });

  // --- FORM SUBMIT (SEND MESSAGE) ---
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = messageInput.value.trim();
    if (!query || state.isGenerating) return;

    // Reset textarea
    messageInput.value = "";
    messageInput.style.height = "auto";
    sendBtn.disabled = true;
    if (clearInputBtn) clearInputBtn.classList.add("hidden");

    // Send query
    await handleUserMessage(query);
  });

  async function handleUserMessage(queryText) {
    if (welcomeHero) {
      welcomeHero.style.display = "none";
    }

    // 1. Ensure active chat session exists in state
    let sess = state.sessions.find(s => s.id === state.currentSessionId);
    if (!sess) {
      const generatedTitle = queryText.length > 28 ? queryText.substring(0, 28) + "..." : queryText;
      sess = {
        id: "sess_" + Date.now(),
        title: generatedTitle,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        mode: state.mode,
        messages: []
      };
      state.sessions.unshift(sess);
      state.currentSessionId = sess.id;
    } else {
      sess.updatedAt = Date.now();
      if ((!sess.title || sess.title === "New Conversation") && (!sess.messages || sess.messages.length === 0)) {
        sess.title = queryText.length > 28 ? queryText.substring(0, 28) + "..." : queryText;
      }
    }

    // 2. Render User Message
    const userTime = formatTime(new Date());
    const userTurn = {
      sender: "user",
      text: queryText,
      time: userTime
    };

    appendMessage(userTurn);
    state.history.push(userTurn);
    sess.messages.push(userTurn);
    saveSessions();
    renderSessionList(historySearchInput ? historySearchInput.value : "");

    // 3. Show Typing Indicator
    if (typingLabel) {
      if (state.mode === "faq") {
        typingLabel.textContent = "Scanning FAQ dataset via TF-IDF...";
      } else if (state.mode === "ai") {
        typingLabel.textContent = "Generating AI Assistant response...";
      } else {
        typingLabel.textContent = "Evaluating via Hybrid Engine...";
      }
    }
    showTyping(true);
    state.isGenerating = true;

    try {
      // Build conversation history for multi-turn context (last 8 turns)
      const historyPayload = state.history.slice(-8).map(turn => ({
        role: turn.sender === "user" ? "user" : "assistant",
        content: turn.text
      }));

      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: queryText,
          mode: state.mode,
          threshold: state.threshold,
          category_filter: state.category,
          history: historyPayload
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const data = await response.json();
      showTyping(false);

      const botTime = formatTime(new Date());

      // Update header latency badge
      if (data.latency_ms !== undefined && headerLatencyBadge) {
        headerLatencyBadge.textContent = `${data.latency_ms}ms`;
      }

      // 4. Render Bot Response
      const botTurn = {
        sender: "bot",
        text: data.response || "No response received.",
        time: botTime,
        promptQuery: queryText,
        meta: {
          source: data.source,
          confidence: data.confidence,
          latency_ms: data.latency_ms,
          matched_faq: data.matched_faq,
          mode: data.mode || state.mode,
          candidates: data.candidates || []
        }
      };

      appendMessage(botTurn);
      state.history.push(botTurn);
      sess.messages.push(botTurn);
      sess.updatedAt = Date.now();
      saveSessions();
      renderSessionList(historySearchInput ? historySearchInput.value : "");

    } catch (err) {
      console.error("Chat API error:", err);
      showTyping(false);
      const errTurn = {
        sender: "bot",
        text: `⚠️ **Connection Error:** Could not connect to the backend server.\n\n*Error details: ${err.message}*`,
        time: formatTime(new Date()),
        promptQuery: queryText,
        meta: {
          source: "System Error",
          confidence: 0,
          latency_ms: 0,
          mode: state.mode
        }
      };
      appendMessage(errTurn);
      state.history.push(errTurn);
      sess.messages.push(errTurn);
      sess.updatedAt = Date.now();
      saveSessions();
      renderSessionList(historySearchInput ? historySearchInput.value : "");
    } finally {
      state.isGenerating = false;
      messageInput.focus();
    }
  }

  // --- RENDER MESSAGE ROW ---
  function appendMessage({ sender, text, time, promptQuery, meta }) {
    if (welcomeHero) {
      welcomeHero.classList.add("hidden");
      welcomeHero.style.display = "none";
    }

    const row = document.createElement("div");
    row.className = `message-row ${sender}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.innerHTML = sender === "user" ? "YOU" : "AI";

    const bubbleWrapper = document.createElement("div");
    bubbleWrapper.className = "message-bubble-wrapper";

    // Meta Header (Sender & Time)
    const metaHeader = document.createElement("div");
    metaHeader.className = "message-meta-header";
    metaHeader.innerHTML = `
      <span class="message-sender">${sender === "user" ? "You" : "Vortex"}</span>
      <span class="message-time">${time}</span>
    `;
    bubbleWrapper.appendChild(metaHeader);

    // Message Bubble Content
    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    if (sender === "user") {
      bubble.textContent = text;
    } else {
      // Parse markdown
      if (typeof marked !== "undefined") {
        bubble.innerHTML = marked.parse(text);
      } else {
        bubble.textContent = text;
      }

      // Highlight code blocks and inject "Copy Code" header
      decorateCodeBlocks(bubble);

      // Render Explainable AI Metadata Badge (for Bot replies)
      if (meta) {
        const xaiCard = createExplainableAiCard(meta, text, promptQuery);
        bubble.appendChild(xaiCard);
      }
    }

    bubbleWrapper.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(bubbleWrapper);

    messagesContainer.appendChild(row);
    scrollToBottom();
  }

  // --- CODE BLOCK DECORATOR (COPY CODE BUTTON + SYNTAX HIGHLIGHT) ---
  function decorateCodeBlocks(container) {
    const preBlocks = container.querySelectorAll("pre");
    preBlocks.forEach((pre) => {
      const code = pre.querySelector("code");
      if (!code) return;

      let lang = "code";
      for (const cls of code.classList) {
        if (cls.startsWith("language-")) {
          lang = cls.replace("language-", "");
          break;
        }
      }

      if (typeof hljs !== "undefined") {
        hljs.highlightElement(code);
      }

      const wrapper = document.createElement("div");
      wrapper.className = "code-block-container";

      const header = document.createElement("div");
      header.className = "code-block-header";
      header.innerHTML = `
        <span>${lang.toUpperCase()}</span>
        <button class="copy-code-btn" type="button">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
          <span>Copy</span>
        </button>
      `;

      const copyBtn = header.querySelector(".copy-code-btn");
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(code.innerText).then(() => {
          copyBtn.innerHTML = `<span>Copied! ✓</span>`;
          setTimeout(() => {
            copyBtn.innerHTML = `
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
              <span>Copy</span>
            `;
          }, 2000);
        });
      });

      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(header);
      wrapper.appendChild(pre);
    });
  }

  // --- EXPLAINABLE AI BADGE CARD WITH RETRY BUTTON ---
  function createExplainableAiCard(meta, answerText, promptQuery) {
    const card = document.createElement("div");
    card.className = "xai-badge-card";

    const badgeRow = document.createElement("div");
    badgeRow.className = "xai-badge-row";

    // 1. Source Pill (Clean Monochrome Editorial Labels)
    const sourcePill = document.createElement("span");
    const src = meta.source || "Unknown";
    if (src.includes("Local FAQ Match")) {
      sourcePill.className = "xai-pill source-local";
      sourcePill.textContent = "LOCAL FAQ";
    } else if (src.includes("Math & Logic Engine")) {
      sourcePill.className = "xai-pill source-math";
      sourcePill.textContent = "MATH ENGINE";
    } else if (src.includes("Utility Engine")) {
      sourcePill.className = "xai-pill source-util";
      sourcePill.textContent = "UTILITY ENGINE";
    } else if (src.includes("Gemini")) {
      sourcePill.className = "xai-pill source-gemini";
      sourcePill.textContent = "GEMINI AI";
    } else if (src.includes("OpenAI")) {
      sourcePill.className = "xai-pill source-openai";
      sourcePill.textContent = "OPENAI";
    } else if (src.includes("Strict FAQ")) {
      sourcePill.className = "xai-pill source-fallback";
      sourcePill.textContent = "OFFLINE MODE";
    } else {
      sourcePill.className = "xai-pill source-fallback";
      sourcePill.textContent = "KNOWLEDGE RETRIEVAL";
    }
    badgeRow.appendChild(sourcePill);

    // 2. Mode Pill
    const modePill = document.createElement("span");
    modePill.className = "xai-pill mode-tag";
    const modeName = (meta.mode === "faq") ? "FAQ ONLY" : (meta.mode === "ai" ? "AI ASSISTANT" : "AUTO");
    modePill.textContent = `MODE: ${modeName}`;
    badgeRow.appendChild(modePill);

    // 3. Confidence Pill
    if (typeof meta.confidence === "number" && meta.confidence > 0) {
      const confPill = document.createElement("span");
      const conf = meta.confidence;
      if (conf >= 50) {
        confPill.className = "xai-pill confidence-high";
      } else {
        confPill.className = "xai-pill confidence-medium";
      }
      confPill.textContent = `Confidence: ${conf.toFixed(1)}%`;
      badgeRow.appendChild(confPill);
    }

    // 4. Latency Pill
    if (meta.latency_ms !== undefined) {
      const latPill = document.createElement("span");
      latPill.className = "xai-pill latency-tag";
      latPill.textContent = `${meta.latency_ms} ms`;
      badgeRow.appendChild(latPill);
    }

    card.appendChild(badgeRow);

    // 5. Matched FAQ Tooltip
    if (meta.matched_faq && meta.matched_faq.question) {
      const matchDetails = document.createElement("div");
      matchDetails.className = "xai-matched-question";
      matchDetails.innerHTML = `
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        <span>Matched FAQ: <em>"${meta.matched_faq.question}"</em></span>
      `;
      card.appendChild(matchDetails);
    }

    // 6. Actions (Copy Answer, Retry, Feedback)
    const actionsRow = document.createElement("div");
    actionsRow.className = "xai-actions";

    // Copy Answer Button
    const copyAnswerBtn = document.createElement("button");
    copyAnswerBtn.className = "xai-btn";
    copyAnswerBtn.type = "button";
    copyAnswerBtn.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
      <span>Copy</span>
    `;
    copyAnswerBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(answerText).then(() => {
        showToast("Answer copied to clipboard!", "success");
      });
    });

    // Retry Button
    const retryBtn = document.createElement("button");
    retryBtn.className = "xai-btn";
    retryBtn.type = "button";
    retryBtn.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>
      <span>Retry</span>
    `;
    retryBtn.addEventListener("click", () => {
      if (promptQuery && !state.isGenerating) {
        showToast("Retrying query...", "info");
        handleUserMessage(promptQuery);
      }
    });

    // Helpful / Feedback Buttons
    const thumbsUpBtn = document.createElement("button");
    thumbsUpBtn.className = "xai-btn";
    thumbsUpBtn.type = "button";
    thumbsUpBtn.innerHTML = `👍 Helpful`;
    thumbsUpBtn.addEventListener("click", () => {
      thumbsUpBtn.style.color = "var(--accent-emerald)";
      showToast("Thank you for your feedback!", "success");
    });

    const thumbsDownBtn = document.createElement("button");
    thumbsDownBtn.className = "xai-btn";
    thumbsDownBtn.type = "button";
    thumbsDownBtn.innerHTML = `👎 Inaccurate`;
    thumbsDownBtn.addEventListener("click", () => {
      thumbsDownBtn.style.color = "var(--accent-rose)";
      showToast("Feedback recorded. You can adjust the threshold slider or add/edit this in Manage FAQs!", "info");
    });

    actionsRow.appendChild(copyAnswerBtn);
    actionsRow.appendChild(retryBtn);
    actionsRow.appendChild(thumbsUpBtn);
    actionsRow.appendChild(thumbsDownBtn);
    card.appendChild(actionsRow);

    return card;
  }

  // --- TYPING INDICATOR TOGGLE ---
  function showTyping(visible) {
    if (visible) {
      typingIndicator.classList.remove("hidden");
      messagesContainer.appendChild(typingIndicator);
      scrollToBottom();
    } else {
      typingIndicator.classList.add("hidden");
    }
  }

  function scrollToBottom() {
    setTimeout(() => {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }, 30);
  }

  function formatTime(d) {
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  // --- CHAT SESSIONS & HISTORY MANAGEMENT ---
  function loadSessionsFromStorage() {
    try {
      const raw = localStorage.getItem("vortex_chat_sessions");
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) return parsed;
      }
    } catch (e) {
      console.warn("Could not load sessions from localStorage:", e);
    }
    return [];
  }

  function saveSessions() {
    try {
      localStorage.setItem("vortex_chat_sessions", JSON.stringify(state.sessions));
    } catch (e) {
      console.warn("Could not save sessions to localStorage:", e);
    }
  }

  function formatSessionTime(ts) {
    if (!ts) return "";
    const date = new Date(ts);
    const now = new Date();
    const diffMs = now - date;
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHours = Math.floor(diffMin / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMin < 1) return "Just now";
    if (diffMin < 60) return `${diffMin}m ago`;
    if (diffHours < 24 && date.getDate() === now.getDate()) {
      return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }
    if (diffDays === 1) return "Yesterday";
    return date.toLocaleDateString([], { month: "short", day: "numeric" });
  }

  function renderSessionList(filterQuery = "") {
    if (!historySessionsList) return;
    historySessionsList.innerHTML = "";

    const query = (filterQuery || "").trim().toLowerCase();
    const filtered = state.sessions.filter(sess => {
      if (!query) return true;
      if (sess.title && sess.title.toLowerCase().includes(query)) return true;
      if (sess.messages && sess.messages.some(m => m.text && m.text.toLowerCase().includes(query))) return true;
      return false;
    });

    if (filtered.length === 0) {
      if (historyEmptyState) {
        historyEmptyState.classList.remove("hidden");
        if (historyEmptyText) {
          historyEmptyText.textContent = query ? "No matching chats" : "No chats yet";
        }
      }
      return;
    }

    if (historyEmptyState) {
      historyEmptyState.classList.add("hidden");
    }

    filtered.forEach(sess => {
      const item = document.createElement("div");
      item.className = "history-session-item" + (sess.id === state.currentSessionId ? " active" : "");
      item.setAttribute("data-session-id", sess.id);

      const mainDiv = document.createElement("div");
      mainDiv.className = "session-main";

      const iconSpan = document.createElement("span");
      iconSpan.className = "session-icon";
      iconSpan.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`;

      const infoDiv = document.createElement("div");
      infoDiv.className = "session-info";

      const titleDiv = document.createElement("div");
      titleDiv.className = "session-title";
      titleDiv.textContent = sess.title || "Conversation";

      const timeDiv = document.createElement("div");
      timeDiv.className = "session-time";
      timeDiv.textContent = formatSessionTime(sess.updatedAt || sess.createdAt);

      infoDiv.appendChild(titleDiv);
      infoDiv.appendChild(timeDiv);

      mainDiv.appendChild(iconSpan);
      mainDiv.appendChild(infoDiv);

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "delete-session-btn";
      delBtn.title = "Delete conversation";
      delBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>`;

      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteSession(sess.id);
      });

      item.appendChild(mainDiv);
      item.appendChild(delBtn);

      item.addEventListener("click", () => {
        loadSession(sess.id);
        if (isMobile()) {
          sidebar.classList.remove("open");
          if (sidebarBackdrop) sidebarBackdrop.classList.remove("active");
        }
      });

      historySessionsList.appendChild(item);
    });
  }

  function startNewChat(focusInput = true) {
    state.currentSessionId = null;
    state.history = [];

    if (messagesContainer) {
      const messages = messagesContainer.querySelectorAll(".message-row");
      messages.forEach(m => m.remove());
    }

    if (welcomeHero) {
      welcomeHero.classList.remove("hidden");
      welcomeHero.style.display = "flex";
    }

    renderSessionList(historySearchInput ? historySearchInput.value : "");

    if (focusInput && messageInput) {
      messageInput.value = "";
      messageInput.style.height = "auto";
      sendBtn.disabled = true;
      messageInput.focus();
    }
  }

  function loadSession(sessionId) {
    const sess = state.sessions.find(s => s.id === sessionId);
    if (!sess) return;

    state.currentSessionId = sess.id;
    state.history = sess.messages ? [...sess.messages] : [];

    if (messagesContainer) {
      const messages = messagesContainer.querySelectorAll(".message-row");
      messages.forEach(m => m.remove());
    }

    if (state.history.length > 0) {
      if (welcomeHero) {
        welcomeHero.classList.add("hidden");
        welcomeHero.style.display = "none";
      }
      state.history.forEach(turn => {
        appendMessage({
          sender: turn.sender,
          text: turn.text,
          time: turn.time || "",
          meta: turn.meta || null
        });
      });
      scrollToBottom();
    } else {
      if (welcomeHero) {
        welcomeHero.classList.remove("hidden");
        welcomeHero.style.display = "flex";
      }
    }

    if (sess.mode && sess.mode !== state.mode) {
      switchMode(sess.mode);
    }

    renderSessionList(historySearchInput ? historySearchInput.value : "");
  }

  function deleteSession(sessionId) {
    state.sessions = state.sessions.filter(s => s.id !== sessionId);
    saveSessions();
    if (state.currentSessionId === sessionId) {
      startNewChat(false);
    } else {
      renderSessionList(historySearchInput ? historySearchInput.value : "");
    }
    showToast("Chat conversation deleted", "info");
  }

  function clearAllHistory() {
    if (!state.sessions || state.sessions.length === 0) {
      showToast("No chat history to clear", "info");
      return;
    }
    if (confirm("Are you sure you want to delete all chat history?")) {
      state.sessions = [];
      saveSessions();
      startNewChat(true);
      showToast("All chat history cleared", "info");
    }
  }

  // --- ROUTING MODE SWITCHER ---
  function switchMode(newMode) {
    state.mode = newMode;

    if (engineModePills) {
      engineModePills.querySelectorAll(".mode-pill, .mode-pill-mini").forEach(p => {
        p.classList.toggle("active", p.getAttribute("data-mode") === newMode);
      });
    }

    if (activeModeBadge) {
      if (newMode === "faq") activeModeBadge.textContent = "FAQ ONLY";
      else if (newMode === "ai") activeModeBadge.textContent = "AI ASSISTANT";
      else activeModeBadge.textContent = "AUTO";
    }

    // Save mode to current session if active
    const sess = state.sessions.find(s => s.id === state.currentSessionId);
    if (sess) {
      sess.mode = newMode;
      saveSessions();
    }
  }

  if (engineModePills) {
    engineModePills.addEventListener("click", (e) => {
      const btn = e.target.closest(".mode-pill, .mode-pill-mini");
      if (!btn) return;
      const rawMode = btn.getAttribute("data-mode");
      let m = "auto";
      if (rawMode === "faq") m = "faq";
      else if (rawMode === "ai" || rawMode === "genai") m = "ai";
      switchMode(m);
      showToast(`Switched routing mode to ${activeModeBadge ? activeModeBadge.textContent : m}`, "info");
    });
  }

  // --- HISTORY EVENT LISTENERS ---
  if (newChatBtn) {
    newChatBtn.addEventListener("click", () => {
      startNewChat(true);
      if (isMobile()) {
        sidebar.classList.remove("open");
        if (sidebarBackdrop) sidebarBackdrop.classList.remove("active");
      }
    });
  }

  if (historySearchInput) {
    historySearchInput.addEventListener("input", (e) => {
      const q = e.target.value;
      if (clearSearchHistoryBtn) {
        clearSearchHistoryBtn.classList.toggle("hidden", !q);
      }
      renderSessionList(q);
    });
  }

  if (clearSearchHistoryBtn) {
    clearSearchHistoryBtn.addEventListener("click", () => {
      if (historySearchInput) {
        historySearchInput.value = "";
        clearSearchHistoryBtn.classList.add("hidden");
        renderSessionList("");
        historySearchInput.focus();
      }
    });
  }

  if (clearAllHistoryBtn) {
    clearAllHistoryBtn.addEventListener("click", clearAllHistory);
  }

  // Clear current chat button in header
  if (clearChatBtn) {
    clearChatBtn.addEventListener("click", () => {
      if (confirm("Are you sure you want to clear this conversation?")) {
        startNewChat(true);
        showToast("Chat conversation cleared", "info");
      }
    });
  }

  // --- INITIALIZE SESSIONS ON LOAD ---
  function initChatHistory() {
    state.sessions = loadSessionsFromStorage();
    if (state.sessions.length > 0 && state.sessions[0].messages && state.sessions[0].messages.length > 0) {
      loadSession(state.sessions[0].id);
    } else {
      renderSessionList();
    }
  }

  initChatHistory();

  // --- EXPORT DROPDOWN ---
  exportDropdownBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    exportDropdownMenu.classList.toggle("show");
  });

  document.addEventListener("click", () => {
    exportDropdownMenu.classList.remove("show");
  });

  // Export TXT
  exportTxtBtn.addEventListener("click", () => {
    if (!state.history.length) {
      showToast("No messages to export yet!", "error");
      return;
    }

    let txtContent = `Vortex AI Chatbot - Conversation Export\n`;
    txtContent += `Date: ${new Date().toLocaleString()}\n`;
    txtContent += `Engine Mode: ${state.mode.toUpperCase()} | Threshold: ${state.threshold}\n`;
    txtContent += `============================================================\n\n`;

    state.history.forEach((m) => {
      const sender = m.sender === "user" ? "USER" : "ASSISTANT";
      const src = m.meta && m.meta.source ? ` [Source: ${m.meta.source}]` : "";
      txtContent += `[${m.time}] ${sender}${src}:\n${m.text}\n\n`;
    });

    downloadBlob(txtContent, "chat_export.txt", "text/plain");
    showToast("Chat exported to TXT", "success");
  });

  // Export JSON
  exportJsonBtn.addEventListener("click", () => {
    if (!state.history.length) {
      showToast("No messages to export yet!", "error");
      return;
    }

    const payload = {
      project: "Vortex AI - Hybrid Chatbot",
      exported_at: new Date().toISOString(),
      engine_settings: {
        mode: state.mode,
        threshold: state.threshold,
        category: state.category
      },
      turns: state.history
    };

    downloadBlob(JSON.stringify(payload, null, 2), "chat_export.json", "application/json");
    showToast("Chat exported to JSON", "success");
  });

  function downloadBlob(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // --- COLLAPSIBLE SIDEBAR & RESPONSIVE DRAWER LOGIC ---
  const isMobile = () => window.innerWidth <= 768;
  const STORAGE_KEY_SIDEBAR = "vortex_sidebar_collapsed";

  // Restore persisted collapsed preference on desktop
  const isInitiallyCollapsed = localStorage.getItem(STORAGE_KEY_SIDEBAR) === "true";
  if (!isMobile() && isInitiallyCollapsed && sidebar) {
    sidebar.classList.add("collapsed");
    if (mobileMenuBtn) {
      mobileMenuBtn.setAttribute("title", "Expand Sidebar");
      mobileMenuBtn.setAttribute("aria-expanded", "false");
    }
  }

  function toggleSidebar() {
    if (!sidebar) return;
    if (isMobile()) {
      const isOpen = sidebar.classList.contains("open");
      if (isOpen) {
        sidebar.classList.remove("open");
        if (sidebarBackdrop) sidebarBackdrop.classList.remove("active");
      } else {
        sidebar.classList.add("open");
        if (sidebarBackdrop) sidebarBackdrop.classList.add("active");
      }
    } else {
      const isCollapsed = sidebar.classList.toggle("collapsed");
      localStorage.setItem(STORAGE_KEY_SIDEBAR, isCollapsed ? "true" : "false");
      if (mobileMenuBtn) {
        mobileMenuBtn.setAttribute("title", isCollapsed ? "Expand Sidebar" : "Collapse Sidebar");
        mobileMenuBtn.setAttribute("aria-expanded", isCollapsed ? "false" : "true");
      }
    }
  }

  if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleSidebar();
    });
  }

  if (sidebarBackdrop) {
    sidebarBackdrop.addEventListener("click", () => {
      if (sidebar) sidebar.classList.remove("open");
      sidebarBackdrop.classList.remove("active");
    });
  }

  // --- MANAGE / ADD / EDIT FAQS MODAL ---
  openManageFaqBtn.addEventListener("click", () => {
    faqManagerModal.classList.remove("hidden");
    loadFaqsList();
  });

  const closeFaqModal = () => {
    faqManagerModal.classList.add("hidden");
    resetFaqForm();
  };

  closeFaqManagerModalBtn.addEventListener("click", closeFaqModal);
  cancelFaqBtn.addEventListener("click", closeFaqModal);

  faqManagerModal.addEventListener("click", (e) => {
    if (e.target === faqManagerModal) {
      closeFaqModal();
    }
  });

  // Tab switching
  tabAddFaqBtn.addEventListener("click", () => {
    tabAddFaqBtn.classList.add("active");
    tabBrowseFaqBtn.classList.remove("active");
    tabContentAdd.classList.add("active");
    tabContentBrowse.classList.remove("active");
  });

  tabBrowseFaqBtn.addEventListener("click", () => {
    tabBrowseFaqBtn.classList.add("active");
    tabAddFaqBtn.classList.remove("active");
    tabContentBrowse.classList.add("active");
    tabContentAdd.classList.remove("active");
    loadFaqsList();
  });

  function resetFaqForm() {
    addFaqForm.reset();
    editFaqId.value = "";
    modalMainTitle.textContent = "FAQ Knowledge Base Manager";
    saveBtnText.textContent = "Save & Retrain Live";
    formNoticeText.innerHTML = "Submitting will immediately append to <code>data/faqs.json</code> and re-fit the TF-IDF vectorizer live in memory!";
  }

  // Submit Add or Edit FAQ Form
  addFaqForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const id = editFaqId.value.trim();
    const q = document.getElementById("newFaqQuestion").value.trim();
    const cat = document.getElementById("newFaqCategory").value;
    const tags = document.getElementById("newFaqTags").value.trim();
    const a = document.getElementById("newFaqAnswer").value.trim();

    if (!q || !a) {
      showToast("Please provide both Question and Answer fields.", "error");
      return;
    }

    const submitBtn = document.getElementById("saveFaqBtn");
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span>Saving & Retraining...</span>`;

    const isEdit = !!id;
    const endpoint = isEdit ? "/edit-faq" : "/add-faq";
    const bodyPayload = isEdit
      ? { faq_id: id, question: q, answer: a, category: cat, tags: tags }
      : { question: q, answer: a, category: cat, tags: tags };

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(bodyPayload)
      });

      const result = await res.json();
      if (!res.ok) {
        throw new Error(result.error || "Failed to save FAQ");
      }

      showToast(isEdit ? "FAQ updated & TF-IDF model re-fitted live!" : "FAQ added & TF-IDF model re-fitted live!", "success");
      closeFaqModal();
      fetchEngineStats();

    } catch (err) {
      console.error("Save FAQ error:", err);
      showToast(`Error: ${err.message}`, "error");
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
        <span id="saveBtnText">${isEdit ? "Update &amp; Retrain" : "Save &amp; Retrain Live"}</span>
      `;
    }
  });

  // --- BROWSE & EDIT FAQS LIST ---
  let loadedFaqs = [];

  async function loadFaqsList() {
    faqsListScrollable.innerHTML = `<p class="loading-text">Loading FAQs from knowledge base...</p>`;
    try {
      const res = await fetch("/faqs");
      const data = await res.json();
      loadedFaqs = data.faqs || [];
      renderFaqsList();
    } catch (err) {
      faqsListScrollable.innerHTML = `<p class="error-text">Failed to load FAQs: ${err.message}</p>`;
    }
  }

  if (faqSearchInput) {
    faqSearchInput.addEventListener("input", () => renderFaqsList());
  }

  if (faqCategoryFilterSelect) {
    faqCategoryFilterSelect.addEventListener("change", () => renderFaqsList());
  }

  function renderFaqsList() {
    const query = (faqSearchInput ? faqSearchInput.value : "").toLowerCase().trim();
    const cat = faqCategoryFilterSelect ? faqCategoryFilterSelect.value : "all";

    let filtered = loadedFaqs;
    if (cat !== "all") {
      filtered = filtered.filter(f => f.category === cat);
    }
    if (query) {
      filtered = filtered.filter(f =>
        (f.question && f.question.toLowerCase().includes(query)) ||
        (f.answer && f.answer.toLowerCase().includes(query)) ||
        (f.tags && f.tags.some(t => t.toLowerCase().includes(query)))
      );
    }

    if (!filtered.length) {
      faqsListScrollable.innerHTML = `<p class="empty-state-text">No FAQs found matching criteria.</p>`;
      return;
    }

    faqsListScrollable.innerHTML = "";
    filtered.slice(0, 50).forEach(faq => {
      const card = document.createElement("div");
      card.className = "faq-item-card";

      const catBadge = document.createElement("span");
      catBadge.className = "faq-cat-badge";
      catBadge.textContent = faq.category || "General";

      const qTitle = document.createElement("h4");
      qTitle.className = "faq-item-question";
      qTitle.textContent = faq.question;

      const aSnippet = document.createElement("p");
      aSnippet.className = "faq-item-answer";
      aSnippet.textContent = faq.answer.length > 140 ? faq.answer.substring(0, 140) + "..." : faq.answer;

      const actions = document.createElement("div");
      actions.className = "faq-item-actions";

      // Edit Button
      const editBtn = document.createElement("button");
      editBtn.className = "btn btn-ghost btn-sm";
      editBtn.innerHTML = `✏️ Edit`;
      editBtn.addEventListener("click", () => {
        openEditFaq(faq);
      });

      // Delete Button
      const delBtn = document.createElement("button");
      delBtn.className = "btn btn-ghost btn-sm text-danger";
      delBtn.innerHTML = `🗑️ Delete`;
      delBtn.addEventListener("click", async () => {
        if (confirm(`Are you sure you want to delete FAQ "${faq.question}"?`)) {
          await deleteFaq(faq.id);
        }
      });

      actions.appendChild(editBtn);
      actions.appendChild(delBtn);

      card.appendChild(catBadge);
      card.appendChild(qTitle);
      card.appendChild(aSnippet);
      card.appendChild(actions);

      faqsListScrollable.appendChild(card);
    });
  }

  function openEditFaq(faq) {
    tabAddFaqBtn.click();
    editFaqId.value = faq.id;
    document.getElementById("newFaqQuestion").value = faq.question || "";
    document.getElementById("newFaqCategory").value = faq.category || "custom";
    document.getElementById("newFaqTags").value = Array.isArray(faq.tags) ? faq.tags.join(", ") : "";
    document.getElementById("newFaqAnswer").value = faq.answer || "";

    modalMainTitle.textContent = `Edit FAQ: "${faq.question.substring(0, 30)}..."`;
    saveBtnText.textContent = "Update & Retrain Live";
    formNoticeText.innerHTML = `Updating this FAQ will modify <code>${faq.id}</code> and re-fit TF-IDF vectorizers in real-time!`;
  }

  async function deleteFaq(faqId) {
    try {
      const res = await fetch("/delete-faq", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ faq_id: faqId })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Delete failed");
      showToast("FAQ deleted & model re-fitted live!", "info");
      loadFaqsList();
      fetchEngineStats();
    } catch (err) {
      showToast(`Error deleting FAQ: ${err.message}`, "error");
    }
  }

  // --- FETCH STATS ---
  async function fetchEngineStats() {
    try {
      const res = await fetch("/stats");
      if (!res.ok) return;
      const data = await res.json();
      if (data.total_faqs && statTotalFaqs) {
        statTotalFaqs.textContent = `${data.total_faqs} items`;
      }
      if (data.model_name && statLlmModel) {
        statLlmModel.textContent = data.model_name;
      }
    } catch (err) {
      console.warn("Failed to fetch engine stats:", err);
    }
  }

  // --- TOAST UTILITY ---
  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;

    let icon = "ℹ️";
    if (type === "success") icon = "✅";
    if (type === "error") icon = "❌";

    toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(10px)";
      toast.style.transition = "all 0.25s ease";
      setTimeout(() => toast.remove(), 250);
    }, 3200);
  }

  // --- BESPOKE SERVICE CARDS & EDITORIAL NAV WIRE-UP ---
  const serviceCardFaq = document.getElementById("serviceCardFaq");
  if (serviceCardFaq) {
    serviceCardFaq.addEventListener("click", () => {
      switchMode("faq");
      showToast("Routing mode switched to FAQ ONLY", "info");
      if (messageInput) messageInput.focus();
    });
  }

  const serviceCardAi = document.getElementById("serviceCardAi");
  if (serviceCardAi) {
    serviceCardAi.addEventListener("click", () => {
      switchMode("ai");
      showToast("Routing mode switched to AI ASSISTANT", "info");
      if (messageInput) messageInput.focus();
    });
  }

  const serviceCardStudio = document.getElementById("serviceCardStudio");
  if (serviceCardStudio) {
    serviceCardStudio.addEventListener("click", () => {
      if (faqManagerModal) {
        faqManagerModal.classList.remove("hidden");
        loadFaqsList();
      }
    });
  }

  const navFaqLink = document.getElementById("navFaqLink");
  if (navFaqLink) {
    navFaqLink.addEventListener("click", (e) => {
      e.preventDefault();
      if (faqManagerModal) {
        faqManagerModal.classList.remove("hidden");
        loadFaqsList();
      }
    });
  }

  document.querySelectorAll(".open-faq-trigger").forEach((trigger) => {
    trigger.addEventListener("click", (e) => {
      e.preventDefault();
      if (faqManagerModal) {
        faqManagerModal.classList.remove("hidden");
        loadFaqsList();
      }
    });
  });
});

