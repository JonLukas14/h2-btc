configfile: "configs/scenario.yaml"

from pathlib import Path

SCENARIO_FILE = config.get("scenario_file", "configs/scenario.yaml")
SCENARIO_NAME = Path(SCENARIO_FILE).stem
RESULTS_DIR = f"results/{SCENARIO_NAME}"

rule all:
    input:
        f"{RESULTS_DIR}/summary.csv"

rule preprocess_inputs:
    input:
        SCENARIO_FILE
    output:
        f"{RESULTS_DIR}/processed_inputs.done"
    log:
        f"{RESULTS_DIR}/logs/preprocess_inputs.log"
    shell:
        r"""
        mkdir -p {RESULTS_DIR}/logs data
        cp {input} configs/scenario.yaml
        python scripts/preprocess_inputs.py > {log} 2>&1
        touch {output}
        """

rule run_model:
    input:
        f"{RESULTS_DIR}/processed_inputs.done",
        SCENARIO_FILE
    output:
        f"{RESULTS_DIR}/network_solved.nc"
    log:
        f"{RESULTS_DIR}/logs/run_model.log"
    shell:
        r"""
        mkdir -p {RESULTS_DIR}/logs {RESULTS_DIR}
        cp {input[1]} configs/scenario.yaml
        python scripts/run_model.py > {log} 2>&1
        cp results/network_solved.nc {output}
        """

rule analyze_results:
    input:
        network="results/{scenario}/network_solved.nc",
        config="configs/scenarios/{scenario}.yaml"
    output:
        summary="results/{scenario}/summary.csv"
    log:
        "results/{scenario}/logs/analyze_results.log"
    shell:
        r"""
        mkdir -p results/{wildcards.scenario}/logs
        python scripts/analyze_results.py \
            --config {input.config} \
            --network {input.network} \
            --outdir results/{wildcards.scenario} \
            > {log} 2>&1
        test -f {output.summary}
        """