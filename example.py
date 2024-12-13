import pandas as pd
from tft_forecaster import TFTForecaster

# # Example: Temperature Forecasting
# def main():
#     # Load and prepare data
#     data = pd.read_csv('datasets/weather_history.csv')

#     # Define features and target
#     target = 'Temperature (C)'
#     time_varying_known_reals = [
#         'Humidity',
#         'Wind Speed (km/h)', 
#         'Wind Bearing (degrees)',
#         'Visibility (km)',
#         'Pressure (millibars)'
#     ]
#     time_varying_unknown_reals = [
#         'Temperature (C)'
#     ]

#     # Initialize forecaster
#     forecaster = TFTForecaster(
#         time_varying_known_reals=time_varying_known_reals,
#         time_varying_unknown_reals=time_varying_unknown_reals,
#         target=target,
#         max_encoder_length=30,  # Look back 30 time steps
#         max_prediction_length=1, # Predict 1 step ahead
#         batch_size=64
#     )

#     # Prepare data
#     forecaster.prep_data(data, test_size=0.02, val_size=0.1)

#     # Train model
#     forecaster.train_model(
#         learning_rate=0.03,
#         hidden_size=32,
#         attention_head_size=4,
#         dropout=0.1,
#         hidden_continuous_size=16,
#         max_epochs=40
#     )

#     # Test model and get MSE
#     mse = forecaster.test_model()

#     # Plot predictions vs actuals
#     forecaster.plot_predictions(y_axis_label="Temperature (C)", plot_title="Temperature Forecasting")

# if __name__ == "__main__":
#     main()


# Example: Web Traffic Forecasting
def main():
    # Load and prepare data
    data = pd.read_csv('datasets/web_traffic.csv')

    # Define features and target
    target = 'hits'
    time_varying_known_reals = [
        'hour',
        'weekday'
    ]
    time_varying_unknown_reals = [
        'hits'
    ]

    # Initialize forecaster
    forecaster = TFTForecaster(
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target=target,
        max_encoder_length=30,  # Look back 30 time steps
        max_prediction_length=1, # Predict 1 step ahead
        batch_size=64
    )

    # Prepare data
    forecaster.prep_data(data, test_size=0.02, val_size=0.1)

    # Train model
    forecaster.train_model(
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        max_epochs=40
    )

    # Test model and get MSE
    mse = forecaster.test_model()

    # Plot predictions vs actuals
    forecaster.plot_predictions(y_axis_label="Hits", plot_title="Web Traffic Forecasting")

if __name__ == "__main__":
    main()