from factor_evaluation import FactorEvaluation, DataService


def ma_deviation_factor(df, ma_period=20):
    ma = df['close'].rolling(ma_period).mean()
    return (df['close']-ma)/ma


evaluator = FactorEvaluation(
    time_periods=['1h'],
    future_return_periods=10,
    factor_func=ma_deviation_factor,
    factor_name='ma_deviation_factor')
results = evaluator.run_full_evaluation()
