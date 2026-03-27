# 🤖 Google Cloud Gemini · Vertex AI Multi-Agent System

> A production-ready multi-agent AI assistant built with **Google Agent Development Kit (ADK)**, **Gemini 2.5 Flash**, and **Vertex AI** — featuring live web search, real-time forex rates, and Wikipedia knowledge retrieval.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Setup from Scratch](#-setup-from-scratch)
  - [1. Google Cloud Project Setup](#1-google-cloud-project-setup)
  - [2. Enable APIs](#2-enable-apis)
  - [3. Authenticate with Google Cloud CLI](#3-authenticate-with-google-cloud-cli)
  - [4. Clone & Install Dependencies](#4-clone--install-dependencies)
  - [5. Configure Environment Variables](#5-configure-environment-variables)
- [Running the Agent](#-running-the-agent)
  - [Run via Google Cloud CLI](#run-via-google-cloud-cli)
  - [Run Locally](#run-locally)
- [Code Walkthrough](#-code-walkthrough)
  - [agent.py — Root Agent](#agentpy--root-agent)
  - [custom\_agents.py — Google Search Sub-Agent](#custom_agentspy--google-search-sub-agent)
  - [custom\_functions.py — Forex Rate Tool](#custom_functionspy--forex-rate-tool)
  - [third\_party\_tools.py — Wikipedia Tool](#third_party_toolspy--wikipedia-tool)
  - [.env — Environment Configuration](#env--environment-configuration)
- [How It Works End-to-End](#-how-it-works-end-to-end)
- [Example Queries](#-example-queries)
- [Troubleshooting](#-troubleshooting)

---

## 🌟 Overview

This project demonstrates how to build a **multi-agent AI system** on Google Cloud that can:

| Capability | Powered By |
|---|---|
| 💱 Real-time currency exchange rates | Custom Python function + Hexarate API |
| 🔍 Live Google Search (current events, weather, business hours) | Google Search sub-agent |
| 📚 Deep historical & cultural knowledge | LangChain Wikipedia tool |
| 🧠 Intelligent routing between tools | Gemini 2.5 Flash on Vertex AI |

The root agent automatically decides which tool(s) to call based on the user's question — no manual routing required.

---

## 🏗️ Architecture

```
User Query
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                      root_agent                         │
│              (Gemini 2.5 Flash · Vertex AI)             │
│                                                         │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────┐ │
│  │  FunctionTool│  │   AgentTool    │  │LangchainTool│ │
│  │  get_fx_rate │  │google_search_  │  │  Wikipedia  │ │
│  │              │  │    agent       │  │             │ │
│  └──────┬───────┘  └───────┬────────┘  └──────┬──────┘ │
└─────────┼──────────────────┼───────────────────┼────────┘
          │                  │                   │
          ▼                  ▼                   ▼
   Hexarate API       Google Search API    Wikipedia API
  (live FX rates)    (real-time results)  (encyclopedic info)
```

---

## ✅ Prerequisites

Before starting, make sure you have the following installed and available:

- **Python 3.11+**
- **pip** (Python package manager)
- **Google Cloud CLI** (`gcloud`) — [Install guide](https://cloud.google.com/sdk/docs/install)
- **A Google Cloud Project** with billing enabled
- **A terminal** (bash/zsh on macOS/Linux or PowerShell on Windows)

---

## 🚀 Setup from Scratch

### 1. Google Cloud Project Setup

Create a new Google Cloud project (or use an existing one):

```bash
# Log in to Google Cloud
gcloud auth login

# Create a new project (skip if using an existing one)
gcloud projects create YOUR_PROJECT_ID --name="Gemini ADK Demo"

# Set the project as your active project
gcloud config set project YOUR_PROJECT_ID
```

> 💡 Replace `YOUR_PROJECT_ID` with your actual project ID (e.g., `my-gemini-agent-001`).

---

### 2. Enable APIs

Enable the required Google Cloud APIs:

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  generativelanguage.googleapis.com \
  cloudresourcemanager.googleapis.com
```

---

### 3. Authenticate with Google Cloud CLI

Set up **Application Default Credentials (ADC)** so the SDK can authenticate automatically:

```bash
gcloud auth application-default login
```

This opens a browser window. Sign in with your Google account that has access to the project.

Verify authentication:

```bash
gcloud auth application-default print-access-token
```

---

### 4. Clone & Install Dependencies

```bash
# Clone the repository
git clone https://github.com/manideepsp/GoogleCloud_Gemini_VertexAI_.git
cd GoogleCloud_Gemini_VertexAI_

# (Recommended) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install all required packages
pip install google-adk \
            google-cloud-aiplatform \
            langchain-community \
            wikipedia \
            requests
```

---

### 5. Configure Environment Variables

Edit the `.env` file at the project root and fill in your values:

```bash
# .env
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT=your-project-id       # ← replace with your project ID
GOOGLE_CLOUD_LOCATION=us-central1          # ← change region if needed
```

> ⚠️ Never commit your `.env` file with real credentials to a public repository.

---

## ▶️ Running the Agent

### Run via Google Cloud CLI

The recommended way to run the ADK web UI is via the `adk web` command. The `imp-code.txt` file captures the full production startup command:

```bash
# Set environment (disable auth for local/dev use)
export ADK_DISABLE_AUTH=true

# Load your .env variables
export $(cat .env | xargs)

# Start the ADK web server
adk web \
  --session_service_uri=memory:// \
  --artifact_service_uri=memory:// \
  --allow_origins="*" \
  --host=0.0.0.0
```

Once started, open your browser and navigate to:

```
http://localhost:8000
```

You will see the ADK web chat interface where you can send messages to the agent.

**Flag explanations:**

| Flag | Description |
|---|---|
| `--session_service_uri=memory://` | Stores session state in memory (no database needed) |
| `--artifact_service_uri=memory://` | Stores artifacts (files, data) in memory |
| `--allow_origins="*"` | Allows requests from any origin (suitable for local dev) |
| `--host=0.0.0.0` | Binds to all network interfaces (accessible on LAN) |

---

### Run Locally

For a quick command-line interaction without the web UI:

```bash
# Load environment variables
export $(cat .env | xargs)

# Run interactively via ADK CLI
adk run GoogleCloud_Gemini_VertexAI_
```

---

## 🧩 Code Walkthrough

### `agent.py` — Root Agent

```python
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.langchain_tool import LangchainTool

from .custom_functions import get_fx_rate
from .custom_agents import google_search_agent
from .third_party_tools import langchain_wikipedia_tool


root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='A helpful assistant for user questions.',
    tools=[
        FunctionTool(get_fx_rate),
        AgentTool(agent=google_search_agent),
        LangchainTool(langchain_wikipedia_tool),
    ]
)
```

**What it does:**

This is the **entry point** of the application. It defines the `root_agent`, which is the top-level orchestrator powered by **Gemini 2.5 Flash** running on **Vertex AI**.

- **`model='gemini-2.5-flash'`** — Uses Google's fast, capable Gemini 2.5 Flash model for reasoning and tool selection.
- **`FunctionTool(get_fx_rate)`** — Wraps the custom Python forex function as a tool the agent can call directly.
- **`AgentTool(agent=google_search_agent)`** — Delegates to a specialized sub-agent for Google Search tasks (agent-as-a-tool pattern).
- **`LangchainTool(langchain_wikipedia_tool)`** — Integrates a LangChain tool, demonstrating cross-framework compatibility.

The agent automatically selects the right tool(s) for each user query using Gemini's built-in reasoning.

---

### `custom_agents.py` — Google Search Sub-Agent

```python
from google.adk.agents import Agent
from google.adk.tools import google_search


google_search_agent = Agent(
    model='gemini-2.5-flash',
    name='google_search_agent',
    description='A search agent that uses google search to get latest information about current events, weather, or business hours.',
    instruction='Use google search to answer user questions about real-time, logistical information.',
    tools=[google_search],
)
```

**What it does:**

This defines a **specialist sub-agent** dedicated to internet searches. Key design decisions:

- **Separation of concerns** — By isolating search logic into its own agent, the root agent stays clean and each specialist can be individually tuned.
- **`instruction`** — Provides the sub-agent with a focused behavioral directive, making it better at search tasks than a general-purpose prompt.
- **`google_search`** — The built-in ADK Google Search tool that queries live web results in real time.
- **Agent-as-a-Tool pattern** — The root agent treats this entire agent as a single "tool", calling it when real-time search is needed.

---

### `custom_functions.py` — Forex Rate Tool

```python
import requests

def get_fx_rate(base: str, target: str):
    """
    Fetches the current exchange rate between two currencies.

    Args:
        base: The base currency (e.g., "SGD").
        target: The target currency (e.g., "JPY").

    Returns:
        The exchange rate information as a json response,
        or None if the rate could not be fetched.
    """
    base_url = "https://hexarate.paikama.co/api/rates/latest"
    api_url = f"{base_url}/{base}?target={target}"

    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
```

**What it does:**

This is a plain Python function that the ADK wraps into a `FunctionTool`. Gemini reads the **docstring** and **type hints** to understand when and how to call it.

- **`base: str`** and **`target: str`** — Gemini extracts these as required parameters from user input (e.g., _"What is the SGD to JPY rate?"_).
- **Hexarate API** — A free, public currency exchange rate API. No API key required.
- **Return value** — Returns raw JSON from the API, which Gemini then formats into a natural language response.
- **ADK auto-wrapping** — When wrapped with `FunctionTool(get_fx_rate)`, ADK generates a JSON schema from the function signature automatically, so Gemini knows the tool's interface.

---

### `third_party_tools.py` — Wikipedia Tool

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

langchain_wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=3000)
)

langchain_wikipedia_tool.description = (
    "Provides deep historical and cultural information on landmarks, concepts, and places."
    "Use this for 'tell me about' or 'what is the history of' type questions."
)
```

**What it does:**

This file showcases **LangChain interoperability** — you can plug any LangChain tool directly into an ADK agent using `LangchainTool`.

- **`WikipediaQueryRun`** — A LangChain tool that queries Wikipedia's public API.
- **`top_k_results=1`** — Returns only the single most relevant article to keep responses focused.
- **`doc_content_chars_max=3000`** — Caps the content at 3000 characters to avoid overwhelming the LLM context window.
- **`tool.description`** — Overrides the default description to guide Gemini on *when* to use this tool (historical/cultural questions) vs. the Google Search agent (current events).

---

### `.env` — Environment Configuration

```bash
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT=apic-track-1-1
GOOGLE_CLOUD_LOCATION=us-central1
```

**What it does:**

This file configures the ADK to use **Vertex AI** as the model backend instead of Google AI Studio.

| Variable | Value | Purpose |
|---|---|---|
| `GOOGLE_GENAI_USE_VERTEXAI` | `1` | Tells ADK to route all Gemini calls through Vertex AI |
| `GOOGLE_CLOUD_PROJECT` | your project ID | The GCP project that will be billed for Vertex AI usage |
| `GOOGLE_CLOUD_LOCATION` | `us-central1` | The Google Cloud region where the Vertex AI endpoint is hosted |

> Using Vertex AI (vs. AI Studio) provides enterprise features: VPC networking, IAM access control, audit logging, and SLA guarantees.

---

## 🔄 How It Works End-to-End

```
1. User sends a message via the ADK web UI or CLI
        │
2. root_agent receives the message
        │
3. Gemini 2.5 Flash analyzes the query and selects the best tool:
        │
        ├─── "What is 100 USD in EUR?"
        │         └─→ FunctionTool → get_fx_rate("USD", "EUR") → Hexarate API
        │
        ├─── "What's the weather in Tokyo tomorrow?"
        │         └─→ AgentTool → google_search_agent → Google Search
        │
        └─── "Tell me about the history of the Eiffel Tower"
                  └─→ LangchainTool → WikipediaQueryRun → Wikipedia API
        │
4. Tool result returned to Gemini
        │
5. Gemini synthesizes a natural language response
        │
6. Response displayed to the user
```

---

## 💬 Example Queries

Try these queries in the ADK web interface:

```
💱 "Convert 500 SGD to JPY"
🔍 "What are the opening hours of the Louvre museum?"
📰 "What happened in tech news today?"
🌤️  "What is the weather like in Sydney right now?"
📚 "What is the history of the Great Wall of China?"
🏛️  "Tell me about the Roman Colosseum"
```

---

## 🛠️ Troubleshooting

**`google.auth.exceptions.DefaultCredentialsError`**
```bash
gcloud auth application-default login
```

**`403 Permission Denied` on Vertex AI**
```bash
# Grant your account the Vertex AI User role
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:YOUR_EMAIL@gmail.com" \
  --role="roles/aiplatform.user"
```

**`ModuleNotFoundError: No module named 'google.adk'`**
```bash
pip install google-adk
```

**`adk` command not found**
```bash
# Make sure your virtual environment is activated
source .venv/bin/activate

# Or install globally
pip install --upgrade google-adk
```

**Port already in use**
```bash
# Run on a different port
adk web --port=8080 --host=0.0.0.0
```

---

## 📄 License

This project is open source. Feel free to use it as a reference for building your own multi-agent systems on Google Cloud.

---

<div align="center">

Built with ❤️ using [Google ADK](https://google.github.io/adk-docs/) · [Gemini](https://deepmind.google/technologies/gemini/) · [Vertex AI](https://cloud.google.com/vertex-ai)

</div>
