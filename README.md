# DE4S_Model
DE4S is a novel time-series forecasting model capable of forecasting
time-variant seasonal structures. DE4S addresses common challenges in
forecasting time-variant seasonal structure by leveraging
HMM's to identify and "memorize" seasonal patterns and their 
relative likelihood.<br> <br>
 ![alt text](https://github.com/kmana1995/DE4S_Model/blob/master/images/Memorized_Structres.jpg?raw=true)<br>
The model has been shown to forecast with competitive accuracy 
when compared to other widely used forecasting methods. See a comparison
of DE4S and the widely used Facebook Prophet model:


Running a forecast is made simple. Required parameters include a dataframe 
containing the dependant variable and date column (df), the
dependant variable name (endog), a date header (date), the initial level 
(level), level smoothing (alpha), trend (trend), and trend smoothing (beta)
parameters. <br><br>

<b> Example:</b> <br>
Initialize:<br>
model = SeasonalSwitchingModel(df, endog, date, level, alpha, trend, beta)<br>

Fit:<br>
fitted_model = model.fit_seasonal_switching_model()<br>

Predict:<br>
fitted_model.predict(n_steps)<br>

Plot seasonal structures:<br>
fitted_model.plot_seasonal_structures()<br><br>


Package dependencies are found in the requirements.txt.

<b>A detailed paper describing the method and mathematics is included in the repository.</b>
