from pathlib import Path

import pandas as pd

from src.optimization.budget_optimizer import optimize_budget


project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "historical_campaigns.csv"

historical_data = pd.read_csv(data_path)

result = optimize_budget(
    historical_data=historical_data,
    total_budget=500_000,
    max_channel_share=0.50,
)

print("\nRecommended allocation:")
print(result["allocation"].round(3).to_string(index=False))

print(f"\nTotal budget: ${result['total_budget']:,.2f}")
print(f"Expected revenue: ${result['expected_revenue']:,.2f}")
print(f"Expected ROAS: {result['expected_roas']:.2f}x")

# Basic validation
assert abs(
    result["allocation"]["allocated_budget"].sum()
    - result["total_budget"]
) < 1e-3

assert (
    result["allocation"]["allocated_budget"] >= -1e-6
).all()

assert (
    result["allocation"]["allocated_budget"]
    <= result["total_budget"] * 0.50 + 1e-3
).all()

assert len(result["allocation"]) == 5

print("\nAll validation checks passed.")
