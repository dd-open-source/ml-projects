## Problem Statement
Given the past data about weather conditions and cloud coverage, predict the cloud coverage for the next two hours at intervals of 30 minutes.

## High Level Approach
We formulate this problem as multi-variable regression wherein the independent variables are the weather conditions and cloud coverage of the past two hours (taken at intervals of ‘x’ minutes) and the dependent variables to be predicted are the cloud coverage values for the next 2 hours (taken at 30 minutes intervals). We use deep neural network as our model.

## Feature Engineering
### Preprocessing
-  Since the training data contains the records from the night time when the cloud coverage values are -1, we remove these records from each day's data by checking for first and last non-negative cloud-coverage values in a day and retaining only those that come within this range.
- We performed feed-forward imputation for other irrelevant values of cloud coverage.
### Feature Engineering
We modified two features in the dataset in order to make them continuous values easily interpretable by a regression model (Reference: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb#scrollTo=K9UVM5Sw9KQN):
- We combined \'Peak Wind Speed @ 6ft [m/s]\' and \'Avg Wind Direction @ 6ft [deg from N]\' to create a vector to represent wind-velocity vector. In this way, we take care of inconsistencies like using wind-direction when wind-speed is 0.
- We broke down \'Azimuth Angle [degrees]\' into its cosine and sine components to avoid the discontinuity encountered while expressing angles in degrees.

### Train and Test datasets Creation
In order to create the training dataset, we took the records given for each day, performed the preprocessing and feature-engineering mentioned above and created X-Y records where X consists of weather conditions features and cloud coverage for two hours (taken at time interval of 'x' minutes) and Y consists of cloud coverage values for next two hours (taken at time interval of 30 minutes). These records were created by striding over all the records at the rate of 1 minutes giving us about 300-600 records per day. All these records were concatenated giving us 166018 records. Further, we applied the same logic to the day-wise records in the test data and we were able to extract 37075 labelled records of which we set aside 20% as the evaluation set and added 80% to the training set, finally giving us 195678 labelled records.
To get the test features, we picked the features of the last 2 hours data from each day of the test dataset sampled at a time interval of 'x' minutes. We performed our prediction on these 300 records.

- Total training records: 195678
- Evaluation records: 7415 records (used for tracking MAE values)
- The training records were split 90-10 for train and validation sets.

[We experimented with multiple values of 'x'. Our final submission uses x=5 minutes]

## Model Architecture
Our model architecture is a deep neural network. We experimented with the following types of layers:
- Combination of LSTM and Dense layers (we hoped the LSTM layers will help with the learning time-series patterns)
- Only Dense layers.

Other hyper parameters chosen:
- Adam Optimizer
- MAE Loss Function
- Early Stopping configured with patience=3 to stop training when validation loss does not decrease.

The architecture of our final model is:
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
reshape_1 (Reshape)          (None, 24, 16)            0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 32)                6272      
_________________________________________________________________
dense_32 (Dense)             (None, 512)               16896     
_________________________________________________________________
dense_33 (Dense)             (None, 512)               262656    
_________________________________________________________________
dense_34 (Dense)             (None, 512)               262656    
_________________________________________________________________
dense_35 (Dense)             (None, 512)               262656    
_________________________________________________________________
dense_36 (Dense)             (None, 512)               262656    
_________________________________________________________________
dense_37 (Dense)             (None, 512)               262656    
_________________________________________________________________
dense_38 (Dense)             (None, 512)               262656    
_________________________________________________________________
dense_39 (Dense)             (None, 512)               262656    
_________________________________________________________________
dense_40 (Dense)             (None, 512)               262656    
_________________________________________________________________
dense_41 (Dense)             (None, 512)               262656    
_________________________________________________________________
dense_42 (Dense)             (None, 512)               262656    
_________________________________________________________________
dense_43 (Dense)             (None, 4)                 2052      
=================================================================
Total params: 2,651,780
Trainable params: 2,651,780
Non-trainable params: 0




