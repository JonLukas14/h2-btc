from pathlib import Path

# -----------------------------------------------------------------------------
# 1. Define available scenarios
# -----------------------------------------------------------------------------
# These names must match the YAML files in configs/scenarios/.
SCENARIOS = [
    "scenario_base",
    "scenario_high_mining",
    "scenario_low_h2_cost",
    "scenario_high_h2_cost",
    "scenario_no_mining",
]

# -----------------------------------------------------------------------------
# 2. Select the scenario to run
# -----------------------------------------------------------------------------
# Pass a scenario name on the command line, for example:
# snakemake --cores 1 -p --config scenario=scenario_base
scenario = config.get("scenario", "scenario_base")
if scenario not in SCENARIOS:
    raise ValueError(f"Unknown scenario '{scenario}'. Available: {SCENARIOS}")

SCENARIO_FILE = f"configs/scenarios/{scenario}.yaml"
RESULTS_DIR = f"results/{scenario}"
DATA_DIR = f"data/{scenario}"


# -----------------------------------------------------------------------------
# 3. Final workflow target
# -----------------------------------------------------------------------------
rule all:
    input:
        f"{RESULTS_DIR}/summary.csv"


# -----------------------------------------------------------------------------
# 4. Preprocess scenario-specific inputs
# -----------------------------------------------------------------------------
rule preprocess_inputs:
    input:
        config=SCENARIO_FILE
    output:
        done=f"{DATA_DIR}/processed_inputs.done"
    log:
        f"{RESULTS_DIR}/logs/preprocess_inputs.log"
    shell:
        r"""
        mkdir -p {RESULTS_DIR}/logs {DATA_DIR}
        python scripts/preprocess_inputs.py \
            --config {input.config} \
            --data-dir {DATA_DIR} \
            > {log} 2>&1
        touch {output.done}
        """


# -----------------------------------------------------------------------------
# 5. Run the PyPSA model
# -----------------------------------------------------------------------------
# This step solves the scenario and writes the solved network to its results folder.
rule run_model:
    input:
        done=f"{DATA_DIR}/processed_inputs.done",
        config=SCENARIO_FILE
    output:
        network=f"{RESULTS_DIR}/network_solved.nc"
    log:
        f"{RESULTS_DIR}/logs/run_model.log"
    shell:
        r"""
        mkdir -p {RESULTS_DIR}/logs {RESULTS_DIR}
        python scripts/run_model.py \
            --config {input.config} \
            --data-dir {DATA_DIR} \
            --output-network {output.network} \
            > {log} 2>&1
        test -f {output.network}
        """

# -----------------------------------------------------------------------------
# 6. Analyze solved results
# -----------------------------------------------------------------------------
rule analyze_results:
    input:
        network=f"{RESULTS_DIR}/network_solved.nc",
        config=SCENARIO_FILE
    output:
        summary=f"{RESULTS_DIR}/summary.csv"
    log:
        f"{RESULTS_DIR}/logs/analyze_results.log"
    shell:
        r"""
        mkdir -p {RESULTS_DIR}/logs {RESULTS_DIR}
        python scripts/analyze_results.py \
            --config {input.config} \
            --network {input.network} \
            --outdir {RESULTS_DIR} \
            > {log} 2>&1
        test -f {output.summary}
        """