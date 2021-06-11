import numpy as  np
import pandas as pd


class Helper:

    def anomaly_filter(self, series, threshold=3.0):
        """
        The anomaly filter uses forward and backward rolling means and standard deviations
        to determine whether or not shocks are structural shifts or non-informative innovations.
        Using forward and backward channels will expose if outliers result in structural
        changes moving forward, and vice-versa.

        Args:
            - series: pd.series, to run anomaly filter over
            - threshold: float, number of std deviations to be considered outlier
        Returns:
            - cleaned_series: pd.series, with anomolies removed
        """
        # calculate forward means
        thirty_period_mean = series.rolling(30, min_periods=0).mean()
        thirty_period_mean.fillna(inplace=True, method='bfill')
        thirty_period_mean[1:] = thirty_period_mean[:-1]

        # calculate reverse means
        reverse_mean = series.sort_index(ascending=False).rolling(30, min_periods=0).mean().tolist()
        reverse_mean.reverse()
        reverse_mean = pd.Series(reverse_mean)
        thirty_period_bwd_mean = reverse_mean
        thirty_period_bwd_mean.fillna(inplace=True, method='ffill')
        thirty_period_bwd_mean[:-1] = thirty_period_bwd_mean[1:]


        fwd_std = (series - thirty_period_mean)**2
        fwd_std = np.sqrt(fwd_std.rolling(30, min_periods=0).mean())
        fwd_std.fillna(inplace=True, method='bfill')
        fwd_std[1:] = fwd_std[:-1]

        bwd_std = (series - thirty_period_bwd_mean)**2
        bwd_std = np.sqrt(bwd_std.sort_index(ascending=False).rolling(30, min_periods=0).mean()).tolist()
        bwd_std.reverse()
        bwd_std = pd.Series(bwd_std)
        bwd_std.fillna(inplace=True, method='bfill')
        bwd_std[1:] = bwd_std[:-1]

        filter_variance = np.where(fwd_std < bwd_std, bwd_std, fwd_std)

        high_filter = thirty_period_mean+filter_variance*threshold
        low_filter = thirty_period_mean-filter_variance*threshold

        series = np.where(series > high_filter, high_filter, series)
        series = np.where(series < low_filter, low_filter, series)

        cleaned_timeseries = series

        return cleaned_timeseries