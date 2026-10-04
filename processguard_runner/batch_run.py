import csv, json, os
from app import run_all, state, raw_results

run_all()

if state.get("status") != "complete":
    raise RuntimeError(state.get("error") or "PROCESSGUARD run failed")

base = os.path.dirname(__file__)
results_path = os.path.join(base, "results.csv")
metrics_path = os.path.join(base, "metrics.json")

with open(results_path, "w", newline="", encoding="utf-8") as f:
    if raw_results:
        writer = csv.DictWriter(f, fieldnames=list(raw_results[0].keys()))
        writer.writeheader()
        writer.writerows(raw_results)

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump({
        "status": state["status"],
        "api_calls": state["api_calls"],
        "completed": state["completed"],
        "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
        "metrics": state["metrics"]
    }, f, indent=2)

print(json.dumps(state["metrics"], indent=2))
print(f"API calls used: {state['api_calls']}")
