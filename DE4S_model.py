import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import norm
from scipy.spatial.distance import cdist


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
            timeseries_df = self.anomaly_filter(self.df.copy())
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
                                        self.fit_seasonal_hmm(seasonal_profiles, seasonal_observations, cycle_states)

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

    def anomaly_filter(self, df):
        """
        The anomaly filter uses forward and backward rolling means and standard deviations
        to determine whether or not shocks are structural shifts or non-informative innovations.
        Using forward and backward channels will expose if outliers result in structural
        changes moving forward, and vice-versa.

        Args:
            - df: dataframe, containing data for fit
        Returns:
            - cleaned_timeseries: dataframe, with anomolies removed
        """
        # calculate forward means
        df['30_PERIOD_FWD_MEAN'] = df[self.endog].rolling(30, min_periods=0).mean().tolist()
        df['30_PERIOD_FWD_MEAN'].fillna(inplace=True, method='bfill')
        df['30_PERIOD_FWD_MEAN'][1:] = df['30_PERIOD_FWD_MEAN'][:-1]

        # calculate reverse means
        reverse_mean = df[self.endog].sort_index(ascending=False).rolling(30, min_periods=0).mean().tolist()
        reverse_mean.reverse()
        df['30_PERIOD_BWD_MEAN'] = reverse_mean
        df['30_PERIOD_BWD_MEAN'].fillna(inplace=True, method='ffill')
        df['30_PERIOD_BWD_MEAN'][:-1] = df['30_PERIOD_BWD_MEAN'][1:]


        df['FWD_STD'] = (df[self.endog] - df['30_PERIOD_FWD_MEAN'])**2
        df['FWD_STD'] = np.sqrt(df['FWD_STD'].rolling(30, min_periods=0).mean())
        df['FWD_STD'].fillna(inplace=True, method='bfill')
        df['FWD_STD'][1:] = df['FWD_STD'][:-1]

        df['BWD_STD'] = (df[self.endog] - df['30_PERIOD_BWD_MEAN'])**2
        bkwd_std = np.sqrt(df['BWD_STD'].sort_index(ascending=False).rolling(30, min_periods=0).mean()).tolist()
        bkwd_std.reverse()
        df['BWD_STD'] = bkwd_std
        df['BWD_STD'].fillna(inplace=True, method='bfill')
        df['BWD_STD'][1:] = df['BWD_STD'][:-1]

        df['FILTER_VARIANCE'] = np.where(df['FWD_STD'] < df['BWD_STD'], df['BWD_STD'], df['FWD_STD'])

        df['HIGH_FILTER'] = df['30_PERIOD_FWD_MEAN']+df['FILTER_VARIANCE']*3
        df['LOW_FILTER'] = df['30_PERIOD_FWD_MEAN']-df['FILTER_VARIANCE']*3

        df[self.endog] = np.where(df[self.endog] > df['HIGH_FILTER'], df['HIGH_FILTER'], df[self.endog])
        df[self.endog] = np.where(df[self.endog] < df['LOW_FILTER'], df['LOW_FILTER'], df[self.endog])

        cleaned_timeseries = df[[self.date_header, self.endog]]

        return cleaned_timeseries

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

        # set initial level and trend
        level = self.initial_level
        trend = self.initial_trend
        projected = [self.initial_level]

        # apply double exponential smoothing to decompose level and trend
        for ind in range(1, len(training_data)):
            # predict time step
            projection = level+trend
            # update level
            level_new = (1 - self.level_smoothing) * (training_data[ind]) + self.level_smoothing * (level + trend)
            # update trend
            trend_new = (1-self.trend_smoothing)*trend+self.trend_smoothing*(level_new-level)
            # append to projected
            projected.append(projection)

            # set to re-iterate
            trend = trend_new
            level = level_new

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
        clusterer = self.run_seasonal_clustering(cluster_fit_data)

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

    def run_seasonal_clustering(self, data):
        """
        This function will run kmeans clustering up to the max_profile number, and select an optimal cluster number.
        The function removes any observations assigned to independant cluster's; considering them outliers
        inhibitive to the intention of the model.

        It will then return a kmeans model fitted to the training data.

        Args:
            - data: df, training data
        Returns:
            - clusterer: fitted KMeans model
        """
        # perform an initial check to ensure we have more obs than possible clusters
        n_rows = len(data)
        if n_rows < self.max_profiles:
            max_clusters = n_rows
        else:
            max_clusters = self.max_profiles

        # create containers for pertinent information
        cluster_number = []
        distortions = []
        rocs = []
        cluster_rocs = []
        rate_of_change = 0

        # iterate through the number of clusters
        n_clusters = 0

        for i in range(max_clusters):
            # set min_obs_threshold, min observations to retain a cluster
            min_obs_threshold = len(data)*0.05
            _pass = False

            # increment the number of clusters and fit
            n_clusters += 1
            clusterer = KMeans(n_clusters = n_clusters)
            clusters = clusterer.fit_predict(data)

            # check if our clusters have <= min_obs_threshold observation, if so this is inviable
            assignment_list = clusters.tolist()
            cluster_instances = [0] * n_clusters
            for x in assignment_list:
                cluster_instances[x] += 1

            for ind in range(n_clusters):
                # if the cluster has < min_obs_threshold, discard the cluster and the observations assigned to it
                if cluster_instances[ind]<=max(min_obs_threshold,1):
                    data = data[[member != ind for member in assignment_list]]
                    assignment_list = (filter((ind).__ne__, assignment_list))
                    _pass = True
            # if _pass has been switched to true, we must re-iterate with the same cluster number,
            # albeit with discarded observations
            if _pass:
                n_clusters -= 1
                continue

            # calculate cluster centers, and distortion
            centers = []
            for cluster in set(clusters):
                center = np.mean(data[clusters == cluster])
                centers.append(center)
            # calculate the distortion
            distortion_new = np.sum(
                np.min(cdist(data, centers, 'euclidean'), axis=1) / data.shape[0]) / n_clusters

            # store information to ID optimal cluster
            if n_clusters >= 2:
                rate_of_change = distortion_new - distortion
                clust_rate_of_change = n_clusters / (n_clusters-1)
                distortion = distortion_new
            if n_clusters == 1:
                distortion = distortion_new
                clust_rate_of_change = 0

            # append to the containers
            cluster_number.append(n_clusters)
            distortions.append(distortion)
            rocs.append(rate_of_change)
            cluster_rocs.append(clust_rate_of_change)

        # calculate the strength of information gain
        strengths = np.divide(rocs, cluster_rocs)
        strengths = np.nan_to_num(strengths, 0)

        # keep either then optimal cluster based on strength, or max_clusters
        if min(strengths) < 0:
            optimal_clusters = min([index for index, strength in enumerate(strengths) if strength < 0]) + 1
        else:
            optimal_clusters = max_clusters

        # fit a final model on the optimal_cluster and set it as the number of profiles
        clusterer = KMeans(n_clusters=optimal_clusters)
        clusterer.fit(data)
        self.n_profiles = optimal_clusters

        return clusterer

    def fit_seasonal_hmm(self, seasonal_profiles, timeseries_observations, cycle_states):
        """
        Wrapper for fitting a HMM to a timeseries of returns.
        This function is effectively defining the hidden state spaces and producing fitted results
        given the timeseries, hidden states, and transition matrix... all calibrated within
        the baum-welch algorithm.

        Args:
            - seasonal_profiles: dict, containing seasons and their parameter sets
            - timeseries_observations: list/array, containing the timeseries
            - cycle_states: list/array, containing the point in a seasonal cycle of an observation
        Returns:
            - fitted_seasonal_values: array, n_obsx1 array of fitted seasonal values
            - seasonality_transition_matrix: array, n_seasons x n_seasons array of transition likelihoods
            - seasonality_features: dict, seasonal features (parameters)
            - observation_probabilities: array, n_obs x n_seasons array of probabilities
        """
        # Run baum-welch to fit HMM using expectation-maximization
        seasonality_transition_matrix, seasonality_features, observation_probabilities = \
            self.run_baum_welch_algorithm(seasonal_profiles, timeseries_observations, cycle_states)
        # Run viterbi to get the MAP forward pass
        state_list, forward_probabilities = self.run_viterbi_algorithm(seasonality_features,
                                                                       seasonality_transition_matrix,
                                                                       timeseries_observations, cycle_states)
        # Fitted values from the HMM
        fitted_seasonal_values = self.fit_values(forward_probabilities, cycle_states, seasonality_transition_matrix,
                                                 seasonality_features)

        return fitted_seasonal_values, seasonality_transition_matrix, seasonality_features, observation_probabilities


    def fit_values(self, observation_probabilities, cycle_states, transition_matrix, state_features):
        """
        This function predicts the value given the state features and observation probabilities.

        Args:
            - observation_probabilities: list/array, the observation-state probabilities
            - cycle_states: list/array, containing the point in a seasonal cycle of an observation
            - transition_matrix: array, transition probabilities from state to state
            - state_features: dict, containing seasons and their parameter sets
        Returns:
            - fitted_values:
        """
        fitted_values = [0]

        state_mu_params = []

        for state in state_features:
            param_set = state_features[state]
            state_mu_param_set = param_set['SEASONALITY_MU'].tolist()
            state_mu_params.append(state_mu_param_set)

        state_mu_params = np.array(state_mu_params).T

        for ind in range(len(observation_probabilities[:-1])):

            cycle = cycle_states[ind+1]

            future_state_probs = sum(np.multiply(observation_probabilities[ind], transition_matrix.T).T) / np.sum(
                np.multiply(observation_probabilities[ind], transition_matrix.T).T)

            weighted_mu = np.sum(np.multiply(future_state_probs, state_mu_params[cycle,:]))

            fitted_values.append(weighted_mu)

        return fitted_values

    def run_viterbi_algorithm(self, state_features, transition_matrix, timeseries_observations, cycle_states):
        """
        This function runs the viterbi algorithm. It considers the most likely forward pass
        of observations running through hidden states with defined features.

        Args:
            - state_features: dict, containing seasons and their parameter sets
            - transition_matrix: array, transition probabilities from state to state
            - timeseries_observations: list/array, containing the timeseries
            - cycle_states: list/array, containing the point in a seasonal cycle of an observation
        Returns:
            - state_list:
            - forward_probabilities:
        """

        # initialized variables
        observation_probabilities = self.create_observation_probabilities(state_features, timeseries_observations,
                                                                          cycle_states)

        alpha = observation_probabilities[0]
        forward_probabilities = [alpha / sum(alpha)]
        forward_trellis = [np.array([alpha] * self.n_profiles) / np.sum(np.array([alpha] * self.n_profiles))]

        # get the probability of a transition p(t|p(o|s))
        for i in range(1, len(observation_probabilities)):
            # the probability of observation k coming from states i:j
            observation_probability = observation_probabilities[i]

            # the probability of moving from state 1ij to state 2ij (given the starting probability alpha)
            state_to_state_probability = np.multiply(alpha, transition_matrix.T).T

            # The probability of observation i coming from state i,j
            forward_probability = np.multiply(observation_probability, state_to_state_probability)
            forward_probability = forward_probability / np.sum(forward_probability)

            # Re-evaluate alpha (probability of being in state i)
            alpha = sum(forward_probability) / np.sum(forward_probability)
            forward_trellis.append(forward_probability)
            forward_probabilities.append(alpha)

        # create empty list to store the states
        state_list = []
        forward_trellis.reverse()

        prev_state = np.where(forward_trellis[0] == np.max(forward_trellis[0]))[1][0]

        # for each step, evaluate the MAP of the state coming from one of the subsequent states
        for member in forward_trellis:
            state = np.where(member == np.max(member[:, prev_state]))[0][0]
            state_list.append(state)
            prev_state = state
        state_list.reverse()

        return state_list, forward_probabilities

    def run_baum_welch_algorithm(self, state_features, timeseries_observations, cycle_states):
        """
        Run a forward backward algorithm on a suspected HMM (Hidden Markov Model). This
        is the step where the parameters for a HMM are fit to the data

        Args:
            - state_features: dict, containing seasons and their parameter sets
            - timeseries_observations:  list/array, containing the timeseries
            - cycle_states: list/array, containing the point in a seasonal cycle of an observation
        Returns:
            - transition_matrix:
            - state_features:
            - observation_probabilities:
        """

        # create the initial transition matrix
        transition_matrix = self.create_transition_matrix()

        # set cumulative probability, this will be used as a breaking criteria for the EM algorithm
        cummulative_probability = np.inf

        for i in range(10):
            # create the observation probabilities given the initial features
            observation_probabilities = self.create_observation_probabilities(state_features, timeseries_observations,
                                                                              cycle_states)

            # run forward and backward pass through
            forward_probabilities, forward_trellis = self.run_forward_pass(transition_matrix, observation_probabilities)
            backward_probabilities, backward_trellis = self.run_backward_pass(transition_matrix,
                                                                              observation_probabilities)
            backward_trellis.reverse()
            backward_probabilities.reverse()

            # update lambda parameter (probability of state i, time j)
            numerator = np.multiply(np.array(forward_probabilities), np.array(backward_probabilities))

            denominator = sum(np.multiply(np.array(forward_probabilities), np.array(backward_probabilities)).T)
            _lambda = []
            for j in range(len(numerator)):
                _lambda.append((numerator[j, :].T / denominator[j]).T)

            # update epsilon parameter (probability of moving for state i to state j)
            numerator = np.multiply(forward_trellis[1:], backward_trellis[:-1])
            epsilon = []
            for g in range(len(numerator)):
                denominator = np.sum(numerator[g, :, :])
                epsilon.append((numerator[g, :, :].T / denominator).T)

            # Update the transition matrix and observation probabilities for the next iteration
            transition_matrix = ((sum(epsilon) / sum(_lambda))).T / sum((sum(epsilon) / sum(_lambda)))

            # Update the state space parameters
            observation_probabilities = _lambda
            state_ind = 0
            for state in state_features:
                param_set = state_features[state]
                state_weight = [0]*len(set(cycle_states))
                state_var = [0]*len(set(cycle_states))
                state_sum = [0]*len(set(cycle_states))

                for ind in range(len(timeseries_observations)):
                    cycle = cycle_states[ind]
                    state_weight[cycle] += _lambda[ind][state_ind]
                    state_sum[cycle] += timeseries_observations[ind] * _lambda[ind][state_ind]
                    state_var[cycle] += _lambda[ind][state_ind] * np.sqrt(
                        (timeseries_observations[ind] - param_set.loc[param_set['CYCLE'] == cycle,
                                                                      'SEASONALITY_MU'].item()) ** 2)

                state_mu_set = np.divide(state_sum, state_weight).tolist()
                state_sigma_set = np.divide(state_var, state_weight).tolist()
                cycle_ind = list(set(cycle_states))
                cycle_ind.sort()
                param_set_new = pd.DataFrame(columns=['CYCLE', 'SEASONALITY_MU', 'SEASONALITY_SIGMA'], data=
                                np.array([cycle_ind, state_mu_set, state_sigma_set]).T)

                state_features.update({state: param_set_new})

                state_ind += 1

            cummulative_probability_new = np.sum(_lambda)
            pcnt_change = (cummulative_probability_new-cummulative_probability)/cummulative_probability
            if pcnt_change < 0.01:
                break
            else:
                cummulative_probability = cummulative_probability_new

        print('Fitted transition matrix: ')
        print(transition_matrix)
        print('Fitted state features: ')
        print(state_features)

        # multiply the probabilities to get the overall probability. Convert to state using MAP
        observation_probabilities = _lambda

        return transition_matrix, state_features, observation_probabilities

    def create_transition_matrix(self):
        """
        This function creates the initial transition matrix for our HMM.
        We initialize the transition probabilities with random descent, however these
        transition probabilities will be adapted as we perform forward and
        backward passes in the broader fwd-bkwd algorithm.

        Returns:
            - transition_matrix: array, initial transition matrix for HMM
        """

        # For each state, create a transition probability for state_i --> state_j
        # We initialize the transition probabilites as decreasing to more distant states
        transition_list = []
        for state in range(self.n_profiles):
            init_probs = [1] * self.n_profiles
            init_mult = [(1 / ((abs(x - state) + 1) * 1.5)) * init_probs[x] for x in range(len(init_probs))]
            state_transition_prob = np.divide(init_mult, sum(init_mult))
            transition_list.append(state_transition_prob)

        transition_matrix = np.array(transition_list)
        print('Initial transition matrix created for {} states: '.format(self.n_profiles))
        print(transition_matrix)
        return transition_matrix

    def run_forward_pass(self, transition_matrix, observation_probabilities):
        """
        The forward pass of a forward backward algorithm. Calculating the forward probabilities

        Args:
            - transition_matrix: array, probability of transitioning from state i to state j
            - observation_probabilities: array, probability of observation i coming from state j
        Returns:
            - forward_results: array, calculated forward probabilities
            - forward_trellis: trellis of stored results from forward pass
        """

        # initialize the variables
        alpha = observation_probabilities[0]
        forward_results = [alpha]
        forward_trellis = [np.array([alpha] * self.n_profiles) / np.sum(np.array([alpha] * self.n_profiles))]

        # get the probability of a transition p(t|p(o|s))
        for i in range(1, len(observation_probabilities)):
            # the probability of observation k coming from states i:j
            observation_probability = observation_probabilities[i]

            # the probability of moving from state 1ij to state 2ij (given the starting probability alpha)
            state_to_state_probability = np.multiply(alpha, transition_matrix.T).T

            # The probability of observation i coming from state i:j
            forward_probability = np.multiply(observation_probability, state_to_state_probability)
            forward_probability = forward_probability / np.sum(forward_probability)

            # Re-evaluate alpha (probability of being in state i at step)
            alpha = sum(forward_probability)
            forward_trellis.append(forward_probability)
            forward_results.append(alpha)

        return forward_results, forward_trellis

    def run_backward_pass(self, transition_matrix, observation_probabilities):
        """
        The backward pass of a forward backward algorithm. Calculating the backward probabilities

        Args:
            - transition_matrix: array, probability of transitioning from state i to state j
            - observation_probabilities: array, probability of observation i coming from state j
        Returns:
            - backward_results: array, calculated backward probabilities
            - backward_trellis: trellis of stored results from backward pass
        """
        # initialize variables
        beta = [1] * self.n_profiles
        backward_results = [beta]
        backward_trellis = [np.array([beta] * self.n_profiles)]

        # get the probability of a transition p(t|p(o|s))
        for i in range(2, len(observation_probabilities) + 1):
            # the probability of observation k coming from states i:j
            observation_probability = observation_probabilities[-i]

            # the probability of moving from state 1ij to state 2ij (given the starting probability alpha)
            state_to_state_probability = np.multiply(beta, transition_matrix)

            # The probability of observation i coming from state i,j
            backward_probability = np.multiply(observation_probability, state_to_state_probability.T).T
            backward_probability = backward_probability / np.sum(backward_probability)

            # Re-evaluate beta (probability of being in state i at step)
            beta = sum(backward_probability.T)
            backward_trellis.append(backward_probability)
            backward_results.append(beta)

        return backward_results, backward_trellis

    def create_observation_probabilities(self, state_features, timeseries_observations, cycle_states):
        """
        Create the observation probabilities given the parameter set of each state

        Args:
            - state_features: dict, containing seasons and their parameter sets
            - timeseries_observations:  list/array, containing the timeseries
            - cycle_states: list/array, containing the point in a seasonal cycle of an observation
        Returns:
             - observation_state_probabilities: array, the state space probabilities
        """

        observation_state_container = []

        for state in state_features:
            parameter_set = state_features[state]

            state_obs_probabilities = []
            for ind in range(len(timeseries_observations)):
                obs = timeseries_observations[ind]
                cycle = cycle_states[ind]

                mu = parameter_set.loc[parameter_set['CYCLE'] == cycle, 'SEASONALITY_MU'].item()
                sigma = parameter_set.loc[parameter_set['CYCLE'] == cycle, 'SEASONALITY_SIGMA'].item()
                obs_state_probability = norm.pdf(obs, loc=mu, scale=sigma)
                state_obs_probabilities.append(obs_state_probability)

            observation_state_container.append(state_obs_probabilities)

        observation_state_probabilities = np.array(observation_state_container).T

        return observation_state_probabilities

if __name__ == '__main__':
    # Running main will run a single fit and predict step on a subset of the "testing_data.csv" data set
    data = pd.read_csv('nfl_timeseries_test_data.csv', parse_dates=['DATE'])
    exog = pd.get_dummies(data['DATE'].dt.weekday, prefix='weekday')
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
    plt.plot(fitted_switching_model.actuals, label='actual')
    plt.plot(fitted_switching_model.fitted_values, label='fit')
    plt.show()
    print(np.sum(abs(fitted_switching_model.impacts)))