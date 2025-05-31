# Jarvis – Modular LLM Assistant Platform

Jarvis is a modular AI assistant platform built with OpenAI GPT, LlamaIndex, Django, and Chainlit.  
It integrates multiple intelligent services including document Q&A, image analysis, speech processing, and web search, designed for flexible prompt control and safe response generation.

---

## 🚀 Features

- 🔎 **RAG-based Document QA** — uses vector store to answer document-based questions  
- 🖼️ **Image Analysis Module** — interprets uploaded images with vision-language prompting  
- 📈 **Stock Market Querying** — retrieves real-time financial data  
- 🔊 **Voice Interaction** — includes ASR (Whisper) for input and TTS (tts-1, Elevenlabs) for output  
- 🌐 **Web Search Integration** — supports fallback search using real-time web API  
- 🛡️ **Response Safety Filtering** — implements rule-based content moderation logic  
- 🧠 **Modular LLM Core** — centralized prompt routing, function calling, and context-aware handling  
- 💾 **Caching Support** — implements persistent response caching to improve reuse and reduce API calls
- ⌨️ **Command Control** — `/rag on`, `/filter level high`, `/web on`, `/cache on`, ...

---

## 🧰 Tech Stack

| Category     | Technologies |
|--------------|--------------|
| **LLM APIs** | OpenAI GPT-4-mini via `openai` |
| **Prompt/Routing** | Function calling |
| **Backend**  | Django |
| **Frontend** | Chainlit |
| **Retrieval** | Manual RAG: document process, vector embedding, retriever construction |
| **Image/Vision** | Vision-language prompt |
| **Stock Module** | `yfinance` API for TW/US real-time stock querying |
| **Speech**   | OpenAI Whisper (ASR), TTS-1 / Elevenlabs (TTS) |
| **Safety**   | Rule-based filtering, prompt-level restriction |
| **Caching**    | JSON-based response cache with hash indexing |
| **Dev Tools**| Python 3.10+, VS Code, .env, Chainlit UI |

---

##  📦 Folder Structure


```
jarvis-assistant/
├── modules/
│ ├── llm/ # Core OpenAI GPT wrapper + prompt routing
│ ├── rag/ # RAG document parsing & vector search
│ ├── vision/ # Image analysis logic
│ ├── speech/ # ASR + TTS services
│ ├── stock/ # yfinance integration
│ ├── safety/ # Output safety filtering
│ ├── web_search/ # External web search fallback module
│ └── cache/ # JSON-based response caching
├── chainlit_app/ # Frontend UI (Chainlit)
├── jarvis_project/ # Django backend (API routes, middleware)
├── config.py # Global config loader (.env parser)
├── .env.example # Environment config template
├── run_jarvis.py # Entry point for launching the assistant
└── requirements.txt # Python dependencies
```

---

## ⚙️ Getting Started

```bash
# 1. Clone the project
git clone https://github.com/your-username/jarvis-assistant.git
cd jarvis-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare environment variables
cp .env.example .env
# Then edit `.env` to include your OpenAI API key

# 4. Run the assistant
python run_jarvis.py
```
