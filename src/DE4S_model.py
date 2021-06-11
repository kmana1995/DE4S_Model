import pandas as pd
import numpy as np
from scipy.stats import norm

from src.models import double_exponential_smoothing as holt
from src.models import k_means
from src.models import seasonal_hmm as hmm
from src.helpers import data_helpers as helper


class SeasonalSwitchingModelResults:

    def __init__(self, df, endog, date_header, trend, level, seasonal_info,
                 impacts, fitted_values, actuals, residuals):
        self.df = df
        self.endog = endog
        self.date_header = date_header
        self.trend = trend
        self.level = level
        self.seasonal_info = seasonal_info
        self.impacts = impacts
        self.fitted_values = fitted_values
        self.actuals = actuals
        self.residuals = residuals

    def plot_seasonal_structures(self):
        """
        Function for plotting the seasonal components.
        """
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        fig, axs = plt.subplots(self.seasonal_info['profile_count'])
        seasonality_features = self.seasonal_info['seasonal_feature_sets']
        i = 1
        day_keys = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday',
                    6: 'Sunday'}
        for state in seasonality_features:
            cycle_points = []
            seasonal_effects = []
            upper_bounds = []
            lower_bounds = []

            state_features = seasonality_features[state]
            for ind, row in state_features.iterrows():
                cycle_points.append(ind)
                point = row['SEASONALITY_MU']
                upper_bound = norm.ppf(0.9, loc=row['SEASONALITY_MU'], scale=row['SEASONALITY_SIGMA'])
                lower_bound = norm.ppf(0.1, loc=row['SEASONALITY_MU'], scale=row['SEASONALITY_SIGMA'])
                seasonal_effects.append(point)
                upper_bounds.append(upper_bound)
                lower_bounds.append(lower_bound)
            weekdays = [day_keys[day] for day in cycle_points]
            axs[i-1].plot(weekdays, seasonal_effects, color='blue')
            axs[i-1].plot(weekdays, upper_bounds, color='powderblue')
            axs[i-1].plot(weekdays, lower_bounds, color='powderblue')
            axs[i-1].fill_between(weekdays, lower_bounds, upper_bounds, color='powderblue')
            axs[i-1].set_title('Seasonal Effects of State {}'.format(i))
            i += 1
        plt.show()

    def predict(self, n_steps, exog=None):
        """
        Predict function for fitted seasonal switching model

        Args:
            - n_steps: int, horizon to forecast over
        Returns:
            - predictions: list, predicted values
        """
        # set up the prediction data frame
        pred_df = self.create_predict_df(n_steps)

        #extract the level
        level = self.level
        trend = self.trend
        predictions = []

        # generate level and trend predictions
        for pred in range(n_steps):
            level = level+trend
            predictions.append(level)

        if self.seasonal_info['level'] is not None:
            # set secondary containers to store predicted seasonal values
            seasonal_values = []
            state_mu_params = []
            cycle_states = pred_df['CYCLE'].tolist()
            seasonality_features = self.seasonal_info['seasonal_feature_sets']
            # extract state parameter sets
            for state in seasonality_features:
                param_set = seasonality_features[state]
                state_mu_param_set = param_set['SEASONALITY_MU'].tolist()
                state_mu_params.append(state_mu_param_set)

            # get the initial state probability (last observation in the fitted model), the transition matrix,
            # and set the mu parameters up for linear algebra use
            state_mu_params = np.array(state_mu_params).T
            observation_probabilities = self.seasonal_info['profile_observation_probabilities']
            initial_prob = observation_probabilities[-1]
            transition_matrix = self.seasonal_info['profile_transition_matrix']

            # predict seasonal steps
            for pred in range(n_steps):
                cycle = cycle_states[pred]

                future_state_probs = sum(np.multiply(initial_prob, transition_matrix.T).T) / np.sum(
                    np.multiply(initial_prob, transition_matrix.T).T)

                weighted_mu = np.sum(np.multiply(future_state_probs, state_mu_params[cycle, :]))

                seasonal_values.append(weighted_mu)
                initial_prob = future_state_probs

        else:
            seasonal_values = [1]*n_steps

        # generate final predictions by multiplying level+trend*seasonality
        predictions = np.multiply(predictions, seasonal_values).tolist()

        if self.impacts is not None and exog is not None:
            predictions += np.dot(exog, self.impacts)

        return predictions

    def create_predict_df(self, n_steps):
        """
        Set up DF to run prediction on

        Args:
            - n_steps: int, horizon to forecast over
        Returns:
            - pred_df: df, prediction horizon
        """
        decomposition_df = self.df[[self.endog, self.date_header]]
        max_date = max(decomposition_df[self.date_header])
        pred_start = max_date + pd.DateOffset(days=1)
        pred_end = pred_start + pd.DateOffset(days=n_steps-1)
        pred_df = pd.DataFrame({self.date_header: pd.date_range(pred_start, pred_end)})

        if self.seasonal_info['level'] == 'weekly':
            pred_df['CYCLE'] = pred_df[self.date_header].dt.weekday

        return pred_df

class SeasonalSwitchingModel:

    def __init__(self,
                 df,
                 endog,
                 date_header,
                 initial_level,
                 level_smoothing,
                 initial_trend,
                 trend_smoothing,
                 seasonality='weekly',
                 max_profiles=10,
                 anomaly_detection=True,
                 exog=None,
                 _lambda=0.1):

        self.df = df
        self.endog = endog
        self.date_header = date_header
        self.initial_level = initial_level
        self.level_smoothing = level_smoothing
        self.initial_trend = initial_trend
        self.trend_smoothing = trend_smoothing
        self.max_profiles = max_profiles
        self.seasonality = seasonality
        self.anomaly_detection = anomaly_detection
        self.exog = exog
        self._lambda = _lambda

    def fit(self):
        '''
        Parent function for fitting the seasonal switching model to a timeseries.

        The seasonal switching model is designed specifically to model timeseries with multiple seasonal
        states which are "hidden" via a HMM, while modeling trend and level components through double exponential
        smoothing.

        Users can also include exogenous regressors as to include impacts of events.

        Returns:
            - SeasonalSwitchingModelResults: a class housing the fitted results
        '''
        # extract the timeseries specifically from the df provided
        timeseries = self.df[self.endog].tolist()

        # pass the time series through an anomaly filter
        if self.anomaly_detection:
            timeseries_df = self.df[[self.endog, self.date_header]].copy()
            timeseries_df[self.endog] = helper.Helper().anomaly_filter(timeseries_df[self.endog])
        else:
            timeseries_df = self.df[[self.endog, self.date_header]].copy()
        # decompose trend and level components using double exponential smoothing
        decomposition_df, trend, level = self.fit_trend_and_level(timeseries_df)

        try:
            # estimate the seasonal profiles to the partially decomposed timeseries via an cluster analysis
            seasonal_profiles = self.estimate_seasonal_profiles(decomposition_df)

            # extract the observed seasonality (decomposed timeseries) and the cycle (a point in the seasonal cycle)
            seasonal_observations = decomposition_df['OBSERVED_SEASONALITY'].tolist()
            cycle_states = decomposition_df['CYCLE'].tolist()

            # fit the seasonal switching HMM
            fitted_seasonal_values, seasonality_transition_matrix, seasonality_features, observation_probabilities = \
                hmm.SeasonalHiddenMarkovModel(self.n_profiles).fit(seasonal_profiles, seasonal_observations,
                                                                   cycle_states)

            # create dict with seasonal components
            seasonal_components = {'level': self.seasonality,
                                   'profile_count': self.n_profiles,
                                   'seasonal_feature_sets': seasonality_features,
                                   'profile_transition_matrix': seasonality_transition_matrix,
                                   'profile_observation_probabilities': observation_probabilities,
                                   'seasonal_fitted_values': fitted_seasonal_values}
        except Exception as e:
            print('Failure fitting seasonal components, reverting to double exponential smoothing')
            print('Error was {}'.format(e))
            fitted_seasonal_values = [1]*len(decomposition_df)
            seasonal_components ={'level': None}

        # perform a final fit as a multiplicative model, between the HMM and the trend/level fit
        fitted_values = np.multiply(decomposition_df['LEVEL_TREND_DECOMPOSITION'], fitted_seasonal_values).tolist()
        residuals = np.subtract(timeseries, fitted_values).tolist()

        if self.exog is not None:
            impacts = self.fit_event_impacts(residuals)
            fitted_values += np.dot(self.exog, impacts)
            residuals = np.subtract(timeseries, fitted_values).tolist()
        else:
            impacts = None

        # store and return class
        results = SeasonalSwitchingModelResults(self.df, self.endog, self.date_header, trend,
                                                level, seasonal_components, impacts,
                                                fitted_values, timeseries, residuals)

        return results

    def fit_event_impacts(self, endog):
        """
        Fit event impacts using ridge regression.

        Args:
            - endog: series, dependent variable
            - _lambda: float, shrinkage parameter (increasing = increased sparsity)
        Returns:
            - coefficients: array, coefficient set
        """
        exog = self.exog.copy()
        coefficients = np.dot(np.dot(np.linalg.inv(np.dot(exog, exog.T)+self._lambda*np.identity(exog.shape[0])),
                              exog).T, endog)
        return coefficients

    def fit_trend_and_level(self, df):
        """
        Fit the trend and level to the timeseries using double exponential smoothing

        Args:
            - df: dataframe, containing data for fit
        Returns:
            - decomposition_df: dataframe, containing level and trend decomposition fit
            - trend: folat, current trend
            - level: float, current level
        """
        # extract the timeseries and begin forming the decomposition data frame
        decomposition_df = df.copy()

        # establish the "grain" (which cycle we're in) and the "cycle" (which point in the seasonal cycle)
        if self.seasonality == 'weekly':
            decomposition_df['GRAIN'] = decomposition_df.index//7
            decomposition_df['ROLLING_GRAIN_MEAN'] = decomposition_df[self.endog].rolling(
                7, min_periods=0).mean().tolist()
            decomposition_df['CYCLE'] = decomposition_df[self.date_header].dt.weekday
        else:
            print("Seasonal profile not set to 'weekly', unable to fit seasona profiling")

        # extract the training timeseries specifically
        training_data = decomposition_df['ROLLING_GRAIN_MEAN']

        projected, trend, level = holt.double_exponential_smoothing(training_data, self.initial_level, self.initial_trend,
                                                      self.level_smoothing, self.trend_smoothing)

        # apply fit to the fit_df
        decomposition_df['LEVEL_TREND_DECOMPOSITION'] = projected

        # get the observed seasonality using the filtered values
        decomposition_df['OBSERVED_SEASONALITY'] = (decomposition_df[self.endog]/
                                                    decomposition_df['LEVEL_TREND_DECOMPOSITION'])

        return decomposition_df, trend, level

    def estimate_seasonal_profiles(self, decomposition_df):
        """
        This function estimates the seasonal profiles within our timeseries. This serves as the initial
        estimates to the state-space parameters fed to the HMM.

        Args:
            - decomposition_df: a decomposed timeseries into level, trend, seasonality
        Returns:
            - seasonal_profiles: dict, a dictionary containing the seasonal profiles and their state space params
        """
        # extract needed vars to create a cluster df
        clustering_df = decomposition_df[['GRAIN', 'CYCLE', 'OBSERVED_SEASONALITY']]

        # do a group by to ensure grain-cycle pairings
        clustering_df = clustering_df.groupby(['GRAIN', 'CYCLE'], as_index=False)['OBSERVED_SEASONALITY'].agg('mean')

        # Normalize the seasonal affects, reducing the impact of relatively large or small values on the search space
        clustering_df['NORMALIZED_SEASONALITY'] = (clustering_df['OBSERVED_SEASONALITY']-
                                                   clustering_df['OBSERVED_SEASONALITY'].mean()
                                                   )/clustering_df['OBSERVED_SEASONALITY'].std()

        # Remove any outliers from the cluster fit df. Given we are attempting to extract common seasonality, outliers
        # simply inhibit the model
        clustering_df['NORMALIZED_SEASONALITY'] = np.where(clustering_df['NORMALIZED_SEASONALITY']<-3, -3,
                                                            clustering_df['NORMALIZED_SEASONALITY'])
        clustering_df['NORMALIZED_SEASONALITY'] = np.where(clustering_df['NORMALIZED_SEASONALITY']>3, 3,
                                  clustering_df['NORMALIZED_SEASONALITY'])

        # pivot the original timeseries to create a feature set for cluster analysis
        cluster_fit_df = clustering_df.pivot(index='GRAIN', columns='CYCLE', values='NORMALIZED_SEASONALITY').reset_index()
        cluster_fit_df.dropna(inplace=True)
        cluster_fit_data = cluster_fit_df.iloc[:, 1:]

        # do the same on the un-processed df, which will be used to ensure classification of all observations
        cluster_pred_df = clustering_df.pivot(index='GRAIN', columns='CYCLE', values='NORMALIZED_SEASONALITY').reset_index()
        cluster_pred_df.dropna(inplace=True)
        cluster_pred_data = cluster_pred_df.iloc[:,1:]

        # Fit the clustering model to extract common seasonal shapes
        clusterer, self.n_profiles = k_means.run_kmeans_clustering(cluster_fit_data, self.max_profiles)

        # apply a final predict to the un-processed df, assigning initial shapes to all observations
        cluster_pred_df['CLUSTER'] = clusterer.predict(cluster_pred_data).tolist()
        cluster_pred_df = cluster_pred_df[['GRAIN', 'CLUSTER']]
        decomposition_df = decomposition_df.merge(cluster_pred_df, how='inner', on='GRAIN')

        # store the initial seasonal profiles (assuming normal distribution of observations) in a dictionary to be used
        # as state-space parameters in the HMM
        seasonal_profiles = {}
        for profile in range(self.n_profiles):
            profile_df = decomposition_df[decomposition_df['CLUSTER'] == profile]
            weekly_profile_mu = profile_df.groupby('CYCLE', as_index=False)['OBSERVED_SEASONALITY'].agg('mean')
            weekly_profile_mu.rename(columns={'OBSERVED_SEASONALITY': 'SEASONALITY_MU'}, inplace=True)
            weekly_profile_sigma = profile_df.groupby('CYCLE', as_index=True
                                                      )['OBSERVED_SEASONALITY'].agg('std').reset_index()
            weekly_profile_sigma.rename(columns={'OBSERVED_SEASONALITY': 'SEASONALITY_SIGMA'}, inplace=True)

            seasonal_profile = weekly_profile_mu.merge(weekly_profile_sigma, how='inner', on='CYCLE')

            seasonal_profiles.update({'PROFILE_{}'.format(profile): seasonal_profile})

        return seasonal_profiles


if __name__ == '__main__':
    # Running main will run a single fit and predict step on a subset of the "testing_data.csv" data set
    data = pd.read_csv('../nfl_timeseries_test_data.csv', parse_dates=['DATE'])
    exog = pd.get_dummies(data['DATE'].dt.weekday, prefix='weekday')
    print(data.loc[data['QUERIES'] == max(data['QUERIES']), 'DATE'])
    exog['super_bowl'] = np.where(data['DATE'].isin([pd.to_datetime('2/8/2016')]), 1, 0)
    exog_2 = pd.get_dummies(data['DATE'].dt.month, prefix='month')
    exog = exog.merge(exog_2, left_index=True, right_index=True)
    data.columns = data.columns.str.upper().str.strip()
    data.sort_values('DATE', inplace=True)
    fit_df = data
    initial_level = fit_df['QUERIES'][:7].mean()
    forecaster = SeasonalSwitchingModel(fit_df, 'QUERIES', 'DATE', initial_level, .2, 0, .2,
                                        max_profiles=10, seasonality='weekly', anomaly_detection=True,
                                        exog=exog, _lambda=25)

    fitted_switching_model = forecaster.fit()
    predictions = fitted_switching_model.predict(10, exog=exog[-10:])
    fitted_switching_model.plot_seasonal_structures()
    import matplotlib.pyplot as plt
    plt.plot(fit_df['DATE'], fitted_switching_model.actuals, label='actual')
    plt.plot(fit_df['DATE'], fitted_switching_model.fitted_values, label='fit')
    plt.show()
    print(fitted_switching_model.impacts)