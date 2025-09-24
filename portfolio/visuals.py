import plotly.graph_objects as go
import pandas as pd

def plot_frontier_and_random(df_random: pd.DataFrame, frontier_vols, frontier_rets, optimal_points: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_random['volatility'], y=df_random['return'], mode='markers',
                             name='Random Portfolios', marker=dict(size=6, opacity=0.6)))
    fig.add_trace(go.Scatter(x=frontier_vols, y=frontier_rets, mode='lines', name='Efficient Frontier'))
    for name, info in optimal_points.items():
        r, v = info['ret'], info['vol']
        fig.add_trace(go.Scatter(x=[v], y=[r], mode='markers+text', name=name, text=[name], textposition='top center',
                                 marker=dict(size=12)))
    fig.update_layout(xaxis_title='Volatility (Std Dev)', yaxis_title='Expected Annual Return')
    return fig

def pie_weights(weights: pd.Series, title: str = 'Weights'):
    fig = go.Figure(go.Pie(labels=weights.index, values=weights.values, hole=0.3))
    fig.update_layout(title=title)
    return fig
