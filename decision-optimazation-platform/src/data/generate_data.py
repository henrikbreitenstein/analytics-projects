import numpy as np
import pandas as pd

from pathlib import Path


def generate_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    channels = {
        "Google Ads": {"base_spend": 10_000, "roas": 3.5},
        "Facebook": {"base_spend": 8_000, "roas": 2.8},
        "LinkedIn": {"base_spend": 6_000, "roas": 2.5},
        "Email": {"base_spend": 2_000, "roas": 5.0},
        "Referral": {"base_spend": 1_500, "roas": 4.0},
    }

    rows = []

    for month in pd.date_range('2025-01-01', periods=12, freq='MS'):
        for channel, params in channels.items():
            spend = params['base_spend'] * rng.uniform(0.8, 1.2)
            impressions = int(spend * rng.uniform(80, 150))
            clicks = int(impressions * rng.uniform(0.01, 0.05))
            conversions = int(clicks * rng.uniform(0.02, 0.10))
            revenue = spend * params['roas'] * rng.uniform(0.8, 1.2)

            rows.append({
                'data' : month,
                'channel' : channel,
                'spend' : round(spend, 2),
                'impressions' : impressions,
                'clicks' : clicks,
                'conversions': conversions,
                'revenue' : round(revenue, 2)
            })

    df = pd.DataFrame(rows)
    df['roas'] = df['revenue'] / df['spend']
    return df

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / 'data' / 'historical_campaigns.csv'

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_data()
    df.to_csv(output_path, index=False)

    print(f'Generated {len(df)} campaign records.')
    print(f'Saved to: {output_path}')
    print(df.groupby('channel')[['spend', 'revenue', 'roas']].mean().round(2))
