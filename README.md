# 📧 Email Assistant Agent

An email assistant agent with authentication, dynamic tools and prompts (middleware), and human-in-the-loop approval before any email is sent, built with LangChain, LangGraph and Google Gemini.

Log in, ask it to check your inbox and reply. It drafts the email, then **stops and waits for you** to approve, edit or reject it before anything is "sent".

> The inbox and sending are simulated: one demo email from Jane, and nothing is really sent. The project is about **controlling** what an agent can do, not about the Gmail API.

## How it works

```
                    ┌──────────────── not logged in ────────────────┐
                    │  prompt: "you can authenticate users"         │
 you ──▶ agent ──▶  │  tools:  authenticate                         │
                    └───────────────────────┬───────────────────────┘
                                            │ correct email + password
                                            ▼  state.authenticated = True
                    ┌────────────────── logged in ──────────────────┐
                    │  prompt: "you can check the inbox and send"   │
                    │  tools:  check_inbox, send_email              │
                    └───────────────────────┬───────────────────────┘
                                            │ send_email(...)
                                            ▼
                              ✋ PAUSE: human approves / edits / rejects
                                            │
                                            ▼
                                       email "sent"
```

| Concept | Where |
|---|---|
| **Custom state**: `authenticated` flag | `AuthenticatedState` |
| **Runtime context**: the valid login | `EmailContext`, read from `.env` |
| **Dynamic tools**: which tools the model sees | `@wrap_model_call` middleware |
| **Dynamic prompt**: system prompt by login status | `@dynamic_prompt` middleware |
| **Human-in-the-loop**: pause before `send_email` | `HumanInTheLoopMiddleware` |

**Model:** Google `gemini-3.1-flash-lite` (free tier).

## Project structure

```
email-assistant-agent/
├── email_agent.py      # The agent + a terminal chat with approve / edit / reject
├── email_agent.ipynb   # Notebook version, step by step
├── langgraph.json      # Config for LangGraph Studio / Agent Chat UI
├── requirements.txt
└── example.env         # Template for your API key and demo login
```

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/zehmbot/email-assistant-agent.git
   cd email-assistant-agent
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Add your API key**: copy `example.env` to `.env` and fill in:
   - `GOOGLE_API_KEY`: free from [Google AI Studio](https://aistudio.google.com/apikey)
   - Optional: `DEMO_EMAIL` / `DEMO_PASSWORD` to change the demo login. The default is `julie@example.com` / `password123`.

## Usage

### Chat in the terminal
```bash
python email_agent.py
```
```
You: julie@example.com, password123
Assistant: You're logged in! You have one email from Jane asking to grab a coffee next week...

You: reply yes to Jane, any draft is fine
============================================================
✋ The assistant wants to send this email:
   To:      jane@example.com
   Subject: Re: Coffee next week
   Body:
     Hi Jane, I'd love to grab a coffee! ...
============================================================
[a]pprove, [e]dit or [r]eject? a

Assistant: Your email to Jane has been sent.
```

### Chat in Agent Chat UI (web app)
1. Start the local server:
   ```bash
   langgraph dev
   ```
2. Open **[agentchat.vercel.app](https://agentchat.vercel.app)** and enter:
   - **Deployment URL:** `http://localhost:2024`
   - **Assistant / Graph ID:** `agent`
   - **LangSmith API key:** leave empty for a local server
3. Log in, ask for a reply, and the approval request shows up in the chat with buttons.

`langgraph dev` also opens **LangGraph Studio**, where you can watch the graph and the state change live.

### Use the notebook
Open `email_agent.ipynb`, select the `.venv` kernel and run the cells.

## Credits

Built while following the [LangChain Academy: LangChain Foundations](https://github.com/langchain-ai/lca-lc-foundations) course (Module 3, Lesson 5). [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui) is an open-source app by LangChain.
