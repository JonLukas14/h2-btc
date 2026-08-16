# -----------------------------------------------------------------------------
# Scenario definition
# -----------------------------------------------------------------------------

SCENARIOS = [
    "s0_reference",
]

# Select scenario from the command line, e.g.:
#
# snakemake --cores 1 -p --config scenario=s0_reference
#
scenario = config.get("scenario", "s0_reference")

if scenario not in SCENARIOS:
    raise ValueError(
        f"Unknown scenario '{scenario}'. "
        f"Available scenarios: {SCENARIOS}"
    )

SCENARIO_FILE = f"configs/scenarios/scenario_{scenario}.yaml"
DATA_DIR = f"data/{scenario}"
RESULTS_DIR = f"results/{scenario}"


# -----------------------------------------------------------------------------
# Final workflow target
# -----------------------------------------------------------------------------

rule all:
    input:
        f"{RESULTS_DIR}/summary.csv"


# -----------------------------------------------------------------------------
# Preprocess scenario-specific inputs
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
# Build and optimize PyPSA network
# -----------------------------------------------------------------------------

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
        mkdir -p {RESULTS_DIR}/logs

        python scripts/run_model.py \
            --config {input.config} \
            --data-dir {DATA_DIR} \
            --output-network {output.network} \
            > {log} 2>&1

        test -f {output.network}
        """


# -----------------------------------------------------------------------------
# Analyze solved network
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
        mkdir -p {RESULTS_DIR}/logs

        python scripts/analyze_results.py \
            --config {input.config} \
            --network {input.network} \
            --outdir {RESULTS_DIR} \
            > {log} 2>&1

        test -f {output.summary}
        """
