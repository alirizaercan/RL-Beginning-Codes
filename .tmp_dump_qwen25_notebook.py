import json
from pathlib import Path

p = Path(
    "/home/aliriza/Documents/RL-Beginning-Codes/fine-tune/training/colab/qwen25_05b/"
    "Qwen_2_5_0_5B_LunarLander_FullFT_and_Scratch_Colab.ipynb"
)
nb = json.loads(p.read_text())
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    print(f"--- CELL {i} ({cell['cell_type']}) ---")
    print("\n".join(src.splitlines()[:120]))
    print()
