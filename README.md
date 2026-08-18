# C-Metric Signal Intelligence Engine

## How to run the web scraper backend

The scraper backend fetches real website content and news headlines so the AI
analyses **confirmed** evidence instead of guessing from training data.

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Start the backend

```
python scraper.py
```

The server starts on **http://localhost:8765**.

### 3. Open the UI

Open `index.html` in your browser. In Step 1:
- Enter your **Groq API key**
- Click **Ping** next to the Scraper backend field to confirm it shows **ONLINE**

### What the scraper does per company

| Step | What happens |
|---|---|
| Homepage fetch | Fetches the root URL, extracts clean text |
| Subpage crawl | Finds and fetches up to 4 high-signal pages (about, news, careers, products) |
| DuckDuckGo news | Searches for recent headlines about the company (no API key needed) |
| DuckDuckGo general | Fetches search snippets (Crunchbase, LinkedIn, Wikipedia hits) |
| LLM hand-off | All scraped text is passed to Agent 1 as PRIMARY EVIDENCE |

### Without the backend

If the scraper is offline, the engine still works — but Agent 1 will label all
claims `[SPECULATIVE]` and scores will be capped accordingly. The UI shows a
⚠ warning in the modal.

### Epistemic scoring rules (Agent 1)

| Score | Meaning |
|---|---|
| 80–100 | Multiple `[CONFIRMED]` signals directly implying outsourcing need |
| 60–79 | 1 `[CONFIRMED]` + supporting `[INFERRED]` signals |
| 40–59 | Mostly `[INFERRED]`, no confirmed outsourcing gap |
| 20–39 | `[SPECULATIVE]` only |
| 0–19 | No meaningful signal found |
