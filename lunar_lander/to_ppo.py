import json
import re

input_path = "action_50_000.json"
output_path = "ppo_action_50_000.json"

with open(input_path, "r") as f:
    data = json.load(f)

records = []
for item in data:
    conversations = item["conversations"]
    human_msg = next(c["value"] for c in conversations if c["from"] == "human")
    gpt_msg = next(c["value"] for c in conversations if c["from"] == "gpt")

    match = re.match(r"(Action:\s*\d+)", gpt_msg)
    solution = match.group(1) if match else gpt_msg

    records.append({
        "messages": [{"role": "user", "content": human_msg}],
        "solution": solution
    })

with open(output_path, "w") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Done. {len(records)} records written to {output_path}")
