import pandas as pd


def summarize(n, cfg):
    weights = (
        n.snapshot_weightings.generators
        .reindex(n.snapshots)
    )

    solar_generation = float(
        (
            n.generators_t.p["solar"]
            * weights
        ).sum()
    )

    wind_generation = float(
        (
            n.generators_t.p["wind"]
            * weights
        ).sum()
    )

    electrolyzer_input = float(
        (
            n.links_t.p0["electrolyzer"]
            * weights
        ).sum()
    )

    hydrogen_output = float(
        (
            -n.links_t.p1["electrolyzer"]
            * weights
        ).sum()
    )

    hydrogen_delivery = float(
        (
            n.generators_t.p["hydrogen_delivery"]
            * weights
        ).sum()
    )

    solar_capacity = float(
        n.generators.at[
            "solar",
            "p_nom_opt",
        ]
    )

    wind_capacity = float(
        n.generators.at[
            "wind",
            "p_nom_opt",
        ]
    )

    electrolyzer_capacity = float(
        n.links.at[
            "electrolyzer",
            "p_nom_opt",
        ]
    )

    solar_available = float(
        (
            n.generators_t.p_max_pu["solar"]
            * solar_capacity
            * weights
        ).sum()
    )

    wind_available = float(
        (
            n.generators_t.p_max_pu["wind"]
            * wind_capacity
            * weights
        ).sum()
    )

    total_generation = (
        solar_generation
        + wind_generation
    )

    total_available = (
        solar_available
        + wind_available
    )

    target_kt = float(
        cfg["hydrogen"]["target_annual_kt_h2"]
    )

    delivered_kg = (
        hydrogen_delivery
        * 1000.0
        / float(
            cfg["hydrogen"][
                "hydrogen_lhv_kwh_per_kg"
            ]
        )
    )

    objective = float(n.objective)

    return {
        "scenario_name": cfg["scenario_name"],
        "solar_capacity_mw": solar_capacity,
        "wind_capacity_mw": wind_capacity,
        "electrolyzer_capacity_mw": electrolyzer_capacity,
        "solar_generation_mwh": solar_generation,
        "wind_generation_mwh": wind_generation,
        "renewable_generation_mwh": total_generation,
        "electrolyzer_input_mwh": electrolyzer_input,
        "hydrogen_output_mwh": hydrogen_output,
        "hydrogen_delivered_mwh": hydrogen_delivery,
        "hydrogen_delivered_kt": delivered_kg / 1_000_000.0,
        "h2_target_annual_kt": target_kt,
        "solar_curtailment_mwh": (
            solar_available
            - solar_generation
        ),
        "wind_curtailment_mwh": (
            wind_available
            - wind_generation
        ),
        "renewable_utilization_rate": (
            total_generation
            / total_available
        ),
        "objective_eur_per_year": objective,
        "lcoh_eur_per_kg_h2": (
            objective
            / delivered_kg
        ),
    }


def write_summary(summary, path):
    pd.DataFrame([summary]).to_csv(
        path,
        index=False,
    )
