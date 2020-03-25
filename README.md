## Timeseries Forecast Project

This is the solution to time series prediction problem. Traditional time series models like ARIMA, ARIMAX, VAR (mostly VAR because of multivariate inputs), could be used to incorporate trend and seasonality. However, these solutions do not really do any kind of continuous input-->output mapping. The given data does not show increasing or decreasing trend and the problem statement of the coding challenge specifically states:
>Given ('low', 'high', 'weight_avg') prices for the past 'm' time steps, predict the next 'n' prices in the sequence.

Ultimately, the provided solution uses regression models mapping *m* inputs to *n* outputs. I have tried to make this code as configurable as possible by using *.yaml* configurations. Please read the config comments and set appropriate config variables to successfully run the application. Also, documentation is generated using Sphinx 1.8.0 and can be read at 'docs\_build\html\index.html' in the parent directory

Two types of tasks can be performed by this solution :
1. Training :   
                load the training data
                perform test_train (validation_train) split.
                train the specific type of model with specific hyperparameters (as selected in config)
                validate the model
                save the model and metadata at specified output directory.
2. Deployment :
                load the test data
                load the desired models
                predict results
                save the test metadata at specified output directory.

## Getting Started

#### Prerequisits
Requires Python 3.x.x
#### Installing
To install all the dependencies, navigate to project folder
Run 'pip install -r requirements.txt'
#### Running
Once all the dependencies are installed, set appropriate configuration at 'config.yaml'
Run 'python main.py'
For extensive documentation please refer to 'docs\_build\html\index.html'
