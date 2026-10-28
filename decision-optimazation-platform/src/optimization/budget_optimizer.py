from llvmlite.ir import Value
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit, minimize

def estimate_response_curves(
    historical_data: pd.DataFrame,
    saturation_multiplier: float = 2.5,
    ) -> dict:

    if saturation_multiplier <= 1:
        raise ValueError('saturation_multiplier must be grater than 1.')

    required = {'channel', 'spend', 'revenue'}

    if not required.issubset(historical_data.columns):
        raise ValueError(f'Data must contain: {required}')

    if historical_data.empty:
        raise ValueError('historical_data cannot be empty')


    if (
        not np.isfinite(historical_data[['spend', 'revenue']]).all().all()
        or (historical_data['spend'] <= 0).any()
        or (historical_data['revenue'] <= 0).any()
    ):
        raise ValueError('Spend and revenue must be finite and positive.')

    curves = {}

    for channel, group in historical_data.groupby('channel'):
        avg_spend = group['spend'].mean()
        avg_revenue = group['revenue'].mean()

        a = avg_revenue * saturation_multiplier
        b = -np.log(1 - avg_revenue/a)/avg_spend

        curves[channel] = {'a': a, 'b': b}

    return curves

def predict_revenue(spend: float, a: float, b: float) -> float:
    return a * (1 - np.exp(-b*spend))

def optimize_budget(
    historical_data: pd.DataFrame,
    total_budget: float = 500_000,
    max_channel_share: float = 0.50,
    ) -> dict:

    if not np.isfinite(total_budget) or total_budget <= 0:
        raise ValueError('total_budget must be finite and positive')

    if not 0 < max_channel_share <= 1:
        raise ValueError('max_channel_share must be in (0, 1].')

    curves = estimate_response_curves(historical_data)
    channels = list(curves)

    if len(channels) * max_channel_share < 1:
        raise ValueError(
            'Infeasible constraints: channel spending limits. Cannot accommodate the entire budget.'
        )

    a = np.array([curves[c]['a'] for c in channels])
    b = np.array([curves[c]['b'] for c in channels])

    def objective(spending):
        revenue = a * (1-np.exp(-b*spending))
        return -np.sum(revenue)

    result = minimize(
        objective,
        x0=np.full(len(channels), total_budget/len(channels)),
        method='SLSQP',
        bounds = [
            (0, total_budget * max_channel_share)
            for _ in channels
        ],
        constraints=[
            {
                'type': 'eq',
                'fun': lambda spending: np.sum(spending) - total_budget,
            }
        ],
        options={'maxiter': 1000, 'ftol': 1e-8},
    )

    if not result.success:
        raise RuntimeError(f'Optimization failed: {result.message}')

    allocation = pd.DataFrame({
        'channel':channels,
        'allocated_budget': result.x,
        'expected_revenue': a * (1-np.exp(-b*result.x))
    })

    allocation['budget_share'] = (
        allocation['allocated_budget'] / total_budget
    )
    allocation['marginal_revenue'] = (
        a * b * np.exp(-b*result.x)
    )
    
    total_revenue = allocation['expected_revenue'].sum()

    return {
        'allocation' : allocation.sort_values(
            'allocated_budget', ascending=False
        ).reset_index(drop=True),
        'total_budget': total_budget,
        'expected_revenue': total_revenue,
        'expected_roas': total_revenue / total_budget,
        'success': result.success
    }

