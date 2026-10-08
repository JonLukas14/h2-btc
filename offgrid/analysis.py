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

    battery_power_mw = 0.0
    battery_energy_mwh = 0.0
    battery_duration_h = 0.0

    if "battery_charger" in n.links.index:
        battery_power_mw = float(
            n.links.at[
                "battery_charger",
                "p_nom_opt",
            ]
        )

        battery_energy_mwh = float(
            n.stores.at[
                "battery_store",
                "e_nom_opt",
            ]
        )

        if abs(battery_power_mw) < 1e-9:
            battery_power_mw = 0.0

        if abs(battery_energy_mwh) < 1e-9:
            battery_energy_mwh = 0.0

        if battery_power_mw > 1e-9:
            battery_duration_h = (
                battery_energy_mwh
                / battery_power_mw
            )

    # --------------------------------------------------------
    # Bitcoin mining
    # --------------------------------------------------------
    bitcoin_capacity_mw = 0.0
    bitcoin_consumption_mwh = 0.0
    bitcoin_utilization_rate = 0.0
    bitcoin_full_load_hours = 0.0

    bitcoin_gross_revenue_eur_per_mwh = 0.0
    bitcoin_gross_revenue_eur_per_year = 0.0
    bitcoin_other_opex_eur_per_year = 0.0
    bitcoin_net_operating_value_eur_per_year = 0.0
    bitcoin_capacity_cost_eur_per_year = 0.0

    if "bitcoin_mining_sink" in n.generators.index:
        bitcoin_cfg = cfg["bitcoin"]

        bitcoin_extendable = bool(
            n.generators.at[
                "bitcoin_mining_sink",
                "p_nom_extendable",
            ]
        )

        if bitcoin_extendable:
            bitcoin_capacity_mw = float(
                n.generators.at[
                    "bitcoin_mining_sink",
                    "p_nom_opt",
                ]
            )
        else:
            bitcoin_capacity_mw = float(
                n.generators.at[
                    "bitcoin_mining_sink",
                    "p_nom",
                ]
            )

        if abs(bitcoin_capacity_mw) < 1e-9:
            bitcoin_capacity_mw = 0.0

        bitcoin_consumption_mwh = float(
            (
                n.generators_t.p[
                    "bitcoin_mining_sink"
                ]
                * weights
            ).sum()
        )

        weighted_hours = float(
            weights.sum()
        )

        if bitcoin_capacity_mw > 1e-9:
            bitcoin_utilization_rate = (
                bitcoin_consumption_mwh
                / (
                    bitcoin_capacity_mw
                    * weighted_hours
                )
            )

            bitcoin_full_load_hours = (
                bitcoin_consumption_mwh
                / bitcoin_capacity_mw
            )

        hashprice = float(
            bitcoin_cfg[
                "hashprice_eur_per_th_day"
            ]
        )

        asic_efficiency = float(
            bitcoin_cfg[
                "asic_efficiency_j_per_th"
            ]
        )

        pue = float(
            bitcoin_cfg.get(
                "pue",
                1.0,
            )
        )

        other_opex = float(
            bitcoin_cfg.get(
                "other_opex_eur_per_mwh",
                0.0,
            )
        )

        mw_per_th_per_s = (
            asic_efficiency
            * pue
            / 1e6
        )

        th_day_per_mwh = (
            (1.0 / 24.0)
            / mw_per_th_per_s
        )

        bitcoin_gross_revenue_eur_per_mwh = (
            hashprice
            * th_day_per_mwh
        )

        bitcoin_gross_revenue_eur_per_year = (
            bitcoin_consumption_mwh
            * bitcoin_gross_revenue_eur_per_mwh
        )

        bitcoin_other_opex_eur_per_year = (
            bitcoin_consumption_mwh
            * other_opex
        )

        bitcoin_net_operating_value_eur_per_year = (
            bitcoin_gross_revenue_eur_per_year
            - bitcoin_other_opex_eur_per_year
        )

        bitcoin_capacity_cost_eur_per_year = (
            bitcoin_capacity_mw
            * float(
                n.generators.at[
                    "bitcoin_mining_sink",
                    "capital_cost",
                ]
            )
        )

    # PyPSA objective already includes BTC operating value
    # through its negative marginal cost.
    net_system_cost_eur_per_year = objective

    # Add BTC operating value back to recover the
    # underlying gross system expenditure.
    gross_system_expenditure_eur_per_year = (
        net_system_cost_eur_per_year
        + bitcoin_net_operating_value_eur_per_year
    )

    return {
        "scenario_name": cfg["scenario_name"],
        "solar_capacity_mw": solar_capacity,
        "wind_capacity_mw": wind_capacity,
        "electrolyzer_capacity_mw": electrolyzer_capacity,
        "battery_power_mw": battery_power_mw,
        "battery_energy_mwh": battery_energy_mwh,
        "battery_duration_h": battery_duration_h,
        "bitcoin_capacity_mw": bitcoin_capacity_mw,
        "bitcoin_consumption_mwh": bitcoin_consumption_mwh,
        "bitcoin_utilization_rate": bitcoin_utilization_rate,
        "bitcoin_full_load_hours": bitcoin_full_load_hours,
        "bitcoin_gross_revenue_eur_per_mwh": bitcoin_gross_revenue_eur_per_mwh,
        "bitcoin_gross_revenue_eur_per_year": bitcoin_gross_revenue_eur_per_year,
        "bitcoin_other_opex_eur_per_year": bitcoin_other_opex_eur_per_year,
        "bitcoin_net_operating_value_eur_per_year": bitcoin_net_operating_value_eur_per_year,
        "bitcoin_capacity_cost_eur_per_year": bitcoin_capacity_cost_eur_per_year,
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
        "gross_system_expenditure_eur_per_year": (
            gross_system_expenditure_eur_per_year
        ),
        "net_system_cost_eur_per_year": (
            net_system_cost_eur_per_year
        ),
        "net_system_cost_eur_per_kg_h2": (
            net_system_cost_eur_per_year
            / delivered_kg
        ),
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
