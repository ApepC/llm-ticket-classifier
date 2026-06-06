"""
classifier.py — LLM-Powered IT Support Ticket Classifier
=========================================================
Author: CS
GitHub: github.com/BobboB

Classifies IT support tickets by category, severity, and
routing destination using few-shot prompting and the
Anthropic Claude API.

Categories: hardware, software, network, identity, cloud
Severity:   low, medium, high, critical
Routing:    help_desk, cloud_team, network_team, security_team

Usage:
    python classifier.py classify --ticket "My laptop won't turn on"
    python classifier.py batch    --input tickets.csv --output results.csv
    python classifier.py evaluate --input labeled.csv
    python classifier.py serve    --port 8000
"""

import argparse, csv, json, os, sys
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import anthropic
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from rich.console import Console
    from rich.table import Table
    from rich import box
except ImportError as e:
    print(f"Missing dependency: {e}\nRun: pip install -r requirements.txt")
    sys.exit(1)

console = Console()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL             = "claude-3-haiku-20240307"   # Fast + cheap for classification
MAX_TOKENS        = 256

# ── Few-shot examples ────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    {
        "ticket": "My laptop screen is cracked and won't display anything",
        "result": {"category": "hardware", "severity": "high",
                   "routing": "help_desk",
                   "reason": "Physical hardware damage requiring in-person repair"}
    },
    {
        "ticket": "Outlook keeps crashing when I open attachments",
        "result": {"category": "software", "severity": "medium",
                   "routing": "help_desk",
                   "reason": "Application crash likely fixable via update or reinstall"}
    },
    {
        "ticket": "I cannot connect to the VPN from home",
        "result": {"category": "network", "severity": "medium",
                   "routing": "network_team",
                   "reason": "VPN connectivity requires network team investigation"}
    },
    {
        "ticket": "My account is locked and I cannot log into any company systems",
        "result": {"category": "identity", "severity": "high",
                   "routing": "security_team",
                   "reason": "Account lockout may indicate security incident"}
    },
    {
        "ticket": "Azure VM in production is down — website is offline",
        "result": {"category": "cloud", "severity": "critical",
                   "routing": "cloud_team",
                   "reason": "Production outage requires immediate cloud team response"}
    },
    {
        "ticket": "Keyboard is making a clicking sound but works fine",
        "result": {"category": "hardware", "severity": "low",
                   "routing": "help_desk",
                   "reason": "Non-critical hardware cosmetic issue"}
    },
    {
        "ticket": "Suspicious email asking for my password — I didn't click anything",
        "result": {"category": "identity", "severity": "high",
                   "routing": "security_team",
                   "reason": "Phishing attempt requires security team review"}
    },
    {
        "ticket": "Need Microsoft Word installed on my new laptop",
        "result": {"category": "software", "severity": "low",
                   "routing": "help_desk",
                   "reason": "Standard software installation request"}
    },
]

SYSTEM_PROMPT = """You are an expert IT support ticket classifier with 10+ years of experience
triaging enterprise help desk tickets. Your job is to classify incoming tickets accurately
so they reach the right team quickly.

You must respond ONLY with a valid JSON object — no preamble, no explanation, no markdown.

Classification rules:
- category: "hardware" | "software" | "network" | "identity" | "cloud"
- severity:  "low" | "medium" | "high" | "critical"
- routing:   "help_desk" | "network_team" | "cloud_team" | "security_team"
- reason:    one sentence explaining the classification

Severity guidelines:
- critical: production down, active security breach, total work stoppage
- high:     significant business impact, account lockout, data loss risk
- medium:   work degraded but possible workaround exists
- low:      minor inconvenience, cosmetic issue, non-urgent request"""


def build_few_shot_prompt(ticket: str) -> str:
    examples = "\n".join([
        f'Ticket: "{ex["ticket"]}"\n'
        f'Response: {json.dumps(ex["result"])}'
        for ex in FEW_SHOT_EXAMPLES
    ])
    return f"""{examples}

Ticket: "{ticket}"
Response:"""


@dataclass
class ClassificationResult:
    ticket:   str
    category: str
    severity: str
    routing:  str
    reason:   str
    model:    str
    confidence: str = "high"   # Claude doesn't return confidence; placeholder


def classify_ticket(ticket: str) -> ClassificationResult:
    """Classify a single ticket using few-shot prompting with Claude."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

    client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": build_few_shot_prompt(ticket)}
        ]
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    parsed = json.loads(raw)
    return ClassificationResult(
        ticket=ticket,
        category=parsed.get("category", "unknown"),
        severity=parsed.get("severity", "medium"),
        routing=parsed.get("routing", "help_desk"),
        reason=parsed.get("reason", ""),
        model=MODEL,
    )


# ── Batch classification ─────────────────────────────────────
def batch_classify(input_path: str, output_path: str) -> None:
    """Classify tickets from a CSV file and write results."""
    results = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tickets = [row["ticket"] for row in reader]

    console.print(f"Classifying {len(tickets)} tickets...")
    for i, ticket in enumerate(tickets, 1):
        console.print(f"  [{i}/{len(tickets)}] {ticket[:60]}...")
        try:
            result = classify_ticket(ticket)
            results.append(asdict(result))
        except Exception as e:
            results.append({"ticket": ticket, "error": str(e)})

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    console.print(f"[green]Results saved to {output_path}[/green]")
    _print_summary(results)


def _print_summary(results: list) -> None:
    table = Table(box=box.ROUNDED, title="[bold]Classification Summary[/bold]")
    table.add_column("Category"); table.add_column("Count", justify="right")
    cats = {}
    for r in results:
        c = r.get("category", "error")
        cats[c] = cats.get(c, 0) + 1
    for cat, count in sorted(cats.items()):
        table.add_row(cat, str(count))
    console.print(table)


# ── Evaluation ───────────────────────────────────────────────
def evaluate(input_path: str) -> None:
    """Evaluate classifier against labeled data. CSV needs: ticket, true_category, true_severity."""
    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cat_correct = sev_correct = total = 0
    errors = []

    for row in rows:
        ticket     = row["ticket"]
        true_cat   = row["true_category"]
        true_sev   = row["true_severity"]
        total     += 1
        try:
            result = classify_ticket(ticket)
            if result.category == true_cat:  cat_correct += 1
            if result.severity == true_sev:  sev_correct += 1
            errors.append({
                "ticket": ticket,
                "pred_cat": result.category, "true_cat": true_cat, "cat_ok": result.category == true_cat,
                "pred_sev": result.severity, "true_sev": true_sev, "sev_ok": result.severity == true_sev,
            })
        except Exception as e:
            console.print(f"[red]Error on '{ticket[:40]}': {e}[/red]")

    console.print(f"\n[bold]Evaluation Results[/bold]")
    console.print(f"Total tickets:      {total}")
    console.print(f"Category accuracy:  {cat_correct/total:.1%}  ({cat_correct}/{total})")
    console.print(f"Severity accuracy:  {sev_correct/total:.1%}  ({sev_correct}/{total})")

    misses = [e for e in errors if not e["cat_ok"]]
    if misses:
        console.print(f"\n[yellow]Category misses ({len(misses)}):[/yellow]")
        for m in misses:
            console.print(f"  Ticket: {m['ticket'][:50]}")
            console.print(f"    Predicted: {m['pred_cat']} | True: {m['true_cat']}")


# ── FastAPI ──────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM IT Support Ticket Classifier",
        description="Classifies IT support tickets using Claude few-shot prompting.",
        version="1.0.0",
        contact={"name": "CS", "url": "https://github.com/BobboB"},
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    class TicketRequest(BaseModel):
        ticket: str

    @app.get("/health")
    def health():
        return {"status": "healthy", "model": MODEL}

    @app.post("/classify")
    def classify_endpoint(req: TicketRequest):
        if not req.ticket.strip():
            raise HTTPException(400, "Ticket text cannot be empty.")
        try:
            result = classify_ticket(req.ticket)
            return asdict(result)
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.get("/categories")
    def categories():
        return {
            "categories": ["hardware","software","network","identity","cloud"],
            "severities": ["low","medium","high","critical"],
            "routing":    ["help_desk","network_team","cloud_team","security_team"],
        }

    return app


# ── CLI ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LLM IT Support Ticket Classifier — Author: CS"
    )
    sub = parser.add_subparsers(dest="command")

    cp = sub.add_parser("classify", help="Classify a single ticket")
    cp.add_argument("--ticket", required=True, help="Ticket description text")

    bp = sub.add_parser("batch", help="Batch classify from CSV")
    bp.add_argument("--input",  required=True, help="Input CSV (column: ticket)")
    bp.add_argument("--output", required=True, help="Output CSV path")

    ep = sub.add_parser("evaluate", help="Evaluate against labeled CSV")
    ep.add_argument("--input", required=True,
                    help="CSV with columns: ticket, true_category, true_severity")

    sp = sub.add_parser("serve", help="Start FastAPI server")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "classify":
        result = classify_ticket(args.ticket)
        console.print(f"\n[bold cyan]Classification Result[/bold cyan]")
        console.print(f"  Category: [bold]{result.category}[/bold]")
        console.print(f"  Severity: [bold]{result.severity}[/bold]")
        console.print(f"  Routing:  [bold]{result.routing}[/bold]")
        console.print(f"  Reason:   {result.reason}")
    elif args.command == "batch":
        batch_classify(args.input, args.output)
    elif args.command == "evaluate":
        evaluate(args.input)
    elif args.command == "serve":
        print(f"Starting classifier API at http://{args.host}:{args.port}")
        uvicorn.run(create_app(), host=args.host, port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
