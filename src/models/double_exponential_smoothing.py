def double_exponential_smoothing(series, initial_level, initial_trend,
                                 level_smoothing, trend_smoothing):
    """
    Fit the trend and level to the timeseries using double exponential smoothing

    Args:
        - series: series, time-series to perform double exponential smoothing on
    Returns:
        - fit: series, double exponential smoothing fit
        - level: float, current level
        - trend: float, current trend
    """
    # set initial level and trend
    level = initial_level
    trend = initial_trend
    fit = [initial_level]

    # apply double exponential smoothing to decompose level and trend
    for ind in range(1, len(series)):
        # predict time step
        projection = level + trend
        # update level
        level_new = (1 - level_smoothing) * (series[ind]) + level_smoothing * (level + trend)
        # update trend
        trend_new = (1 - trend_smoothing) * trend + trend_smoothing * (level_new - level)
        # append to projected
        fit.append(projection)

        # set to re-iterate
        trend = trend_new
        level = level_new

    return fit, trend, level