import subprocess
import sys
from pathlib import Path
from build_network import build_test_network

BASE_DIR = Path(__file__).resolve().parent.parent

def run_preprocessing():
    script = BASE_DIR / "scripts" / "preprocess_inputs.py"
    subprocess.run([sys.executable, str(script)], check=True)

def main():
    run_preprocessing()
    # then continue with your existing build / solve workflow

n = build_test_network()
status, condition = n.optimize(solver_name="highs")

print("Status:", status)
print("Termination:", condition)
print("Objective value:", n.objective)

print("\nGenerators:")
print(n.generators[["carrier", "p_nom_opt"]])

print("\nLinks:")
print(n.links[["carrier", "p_nom_opt"]])

print("\nStores:")
print(n.stores[["carrier", "e_nom_opt"]])