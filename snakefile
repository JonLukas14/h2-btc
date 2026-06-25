configfile: "configs/scenario.yaml"

rule all:
    input:
        "results/summary.csv"

rule preprocess_inputs:
    output:
        "data/processed_inputs.done"
    log:
        "results/logs/preprocess_inputs.log"
    shell:
        r"""
        mkdir -p results/logs data
        python scripts/preprocess_inputs.py > {log} 2>&1
        touch {output}
        """

rule run_model:
    input:
        "data/processed_inputs.done",
        "configs/scenario.yaml"
    output:
        "results/network_solved.nc"
    log:
        "results/logs/run_model.log"
    shell:
        r"""
        mkdir -p results/logs results
        python scripts/run_model.py > {log} 2>&1
        test -f {output}
        """

rule analyze_results:
    input:
        "results/network_solved.nc",
        "configs/scenario.yaml"
    output:
        "results/summary.csv"
    log:
        "results/logs/analyze_results.log"
    shell:
        r"""
        mkdir -p results/logs results
        python scripts/analyze_results.py > {log} 2>&1
        test -f {output}
        """