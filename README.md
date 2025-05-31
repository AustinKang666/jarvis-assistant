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
- 💾 **Memory Support** — persistent user history and parameterized response memory (via config settings)  
- ⌨️ **Command Control** — e.g., `/filter level high`, `/web`, `/upload`, `/rag off`  

---

## 🧰 Tech Stack

| Category     | Technologies |
|--------------|--------------|
| **LLM APIs** | OpenAI GPT-4-mini via `openai` |
| **Retrieval** | Manual RAG: document process, vector embedding, retriever construction |
| **Backend**  | Django |
| **Frontend** | Chainlit |
| **Speech**   | OpenAI Whisper (ASR), TTS-1 / Elevenlabs (TTS) |
| **Safety**   | Rule-based filtering, prompt-level restriction |
| **Dev Tools**| Python 3.10+, VS Code, .env, Chainlit CLI |

---

## ⚙️ Getting Started

```bash
# Clone the project
git clone https://github.com/your-username/jarvis-assistant.git
cd jarvis-assistant

# Install dependencies
pip install -r requirements.txt