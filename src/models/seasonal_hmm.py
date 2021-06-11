import numpy as np
from scipy.stats import norm
import pandas as pd


class SeasonalHiddenMarkovModel:

    def __init__(self, n_profiles):
        self.n_profiles = n_profiles

    def fit(self, seasonal_profiles, timeseries_observations, cycle_states):
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

    @staticmethod
    def fit_values(observation_probabilities, cycle_states, transition_matrix, state_features):
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

    @staticmethod
    def create_observation_probabilities(state_features, timeseries_observations, cycle_states):
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