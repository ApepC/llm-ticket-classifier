# LLM-Powered IT Support Ticket Classifier

**Author: CS** | [GitHub](https://github.com/ApepC)

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Anthropic](https://img.shields.io/badge/Claude-Few--Shot_Classification-orange?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge&logo=fastapi)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI-2088FF?style=for-the-badge&logo=githubactions)

Classifies IT support tickets by **category**, **severity**, and **routing destination**
using few-shot prompting with the Anthropic Claude API. Built from real IT support experience.

---

## Classification Schema

| Field | Values |
|---|---|
| **Category** | `hardware` · `software` · `network` · `identity` · `cloud` |
| **Severity** | `low` · `medium` · `high` · `critical` |
| **Routing** | `help_desk` · `network_team` · `cloud_team` · `security_team` |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/ApepC/llm-ticket-classifier.git
cd llm-ticket-classifier

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env

# 4. Classify a single ticket
python src/classifier.py classify \
  --ticket "Azure production database is down, website is showing 500 errors"

# 5. Batch classify from CSV
python src/classifier.py batch \
  --input data/sample_tickets.csv \
  --output results.csv

# 6. Evaluate accuracy against labeled data
python src/classifier.py evaluate \
  --input data/labeled_tickets.csv

# 7. Start REST API
python src/classifier.py serve
```

---

## Example Output

```bash
$ python src/classifier.py classify \
    --ticket "Azure production VM is down — website offline"

Classification Result
  Category: cloud
  Severity: critical
  Routing:  cloud_team
  Reason:   Production outage requires immediate cloud team response
```

---

## API

```bash
# Start server
python src/classifier.py serve --port 8000

# Classify via API
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"ticket": "I cannot log in — account says locked"}'
```

**Response:**
```json
{
  "ticket": "I cannot log in — account says locked",
  "category": "identity",
  "severity": "high",
  "routing": "security_team",
  "reason": "Account lockout may indicate security incident requiring investigation",
  "model": "claude-3-haiku-20240307"
}
```

---

## Key AI Concepts Demonstrated

- **Few-shot prompting** — 8 labeled examples guide classification consistency
- **Chain-of-thought** — model returns `reason` field explaining its decision
- **Structured output** — JSON-only responses parsed and validated
- **Prompt engineering** — system prompt defines schema, rules, and severity guidelines
- **Model evaluation** — `evaluate` command calculates category and severity accuracy
- **Batch processing** — CSV ingestion and output for bulk classification workflows

---

## Project Structure

```
llm-ticket-classifier/
├── src/
│   └── classifier.py          # Main: classify, batch, evaluate, serve
├── data/
│   ├── sample_tickets.csv     # Sample tickets for testing
│   └── labeled_tickets.csv    # Labeled data for evaluation
├── .github/workflows/
│   └── ci.yml                 # Lint + validation CI
├── requirements.txt
├── .env.example
└── README.md
```

---

*Built by CSolo —*
