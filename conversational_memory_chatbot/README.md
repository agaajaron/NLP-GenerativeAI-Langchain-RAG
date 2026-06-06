# Conversational Memory Chatbot

> **Demonstrates:** `RunnableWithMessageHistory` · LCEL · session-scoped in-memory chat history · GPT-4

## What it does

A minimal chatbot that remembers what you said earlier in the conversation. Each session is keyed by a `session_id` so multiple independent conversations can run without cross-contamination.

## Project structure

| File | Purpose |
|------|---------|
| `01_setup.py` | LLM init, prompt template, session history store, `chatbot` runnable |
| `02_chatbot.py` | Scripted demo + optional interactive REPL (`--interactive`) |

## Quick start

```bash
pip install -r requirements.txt
cp ../.env.example ../.env   # fill in OPENAI_API_KEY
python 02_chatbot.py          # run scripted demo
python 02_chatbot.py -i       # run demo then open live chat
```

## Architecture

```
User message
    ↓
ChatPromptTemplate
  ├─ system: "You are a helpful assistant."
  ├─ MessagesPlaceholder("history")   ← injected from store
  └─ human: {input}
    ↓  LCEL
ChatOpenAI (GPT-4)
    ↓
RunnableWithMessageHistory
  └─ session store  {session_id: ChatMessageHistory}
```

Each call automatically appends the human message and AI reply to the session's `ChatMessageHistory`, so follow-up questions have full context.

## Tech stack

![LangChain](https://img.shields.io/badge/LangChain-0.2-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green)
![LCEL](https://img.shields.io/badge/LCEL-RunnableWithMessageHistory-purple)
