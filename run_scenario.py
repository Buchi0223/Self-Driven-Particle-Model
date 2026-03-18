"""
シナリオ実行エントリポイント

Usage:
    uv run python run_scenario.py [scenario_name]
    uv run python run_scenario.py all

Available scenarios:
    accel_field   — Step 9:  加速度ベクトル場 (Fig.3)
    single_pass   — Step 10: 単独追い越し
    pass_between  — Step 11: 2台の間を通過 (Fig.5a)
    circumvent    — Step 12: 2台を迂回 (Fig.5b)
    all           — 全シナリオ実行
"""

import sys


def main():
    scenarios = {
        "accel_field": "scenarios.scenario_accel_field",
        "single_pass": "scenarios.scenario_single_pass",
        "pass_between": "scenarios.scenario_pass_between",
        "circumvent": "scenarios.scenario_circumvent",
    }

    args = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    if "all" in args:
        targets = list(scenarios.keys())
    else:
        targets = args

    for name in targets:
        if name not in scenarios:
            print(f"Unknown scenario: {name}")
            print(f"Available: {', '.join(scenarios.keys())}, all")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        import importlib
        mod = importlib.import_module(scenarios[name])
        mod.main()

    print(f"\nAll done. Check output/ directory for results.")


if __name__ == "__main__":
    main()
