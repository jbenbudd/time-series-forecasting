import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.loggers import TensorBoardLogger
import lightning.pytorch as pl
from sklearn.metrics import mean_squared_error


class TFTForecaster:
    def __init__(self, features, target, max_encoder_length=30, max_prediction_length=1, batch_size=64):
        self.features = features
        self.target = target
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.batch_size = batch_size
        self.model = None
        self.trainer = None

    def prep_data(self, data, test_size=0.2, val_size=0.1, static_categoricals=None, time_varying_known_reals=None, time_varying_unknown_reals=None, group_id="default_group"):
        # Add required columns
        data = data.copy()
        data["time_idx"] = range(len(data))  # Add a time index column
        data["group_id"] = group_id if group_id == "default_group" else data[group_id]

        # Determine lengths of validation and test sets
        test_length = int(len(data) * test_size)
        val_length = int(len(data) * val_size)

        # Ensure validation + test length is not greater than total data length
        if val_length + test_length > len(data):
            raise ValueError("Validation and test set sizes exceed dataset length.")

        # Split data into training, validation, and test sets
        train_data = data.iloc[: -(val_length + test_length)]
        val_data = data.iloc[-(val_length + test_length):-test_length]
        test_data = data.iloc[-test_length:]

         # Save test dates for plotting
        self.test_dates = test_data["date"].values

        # Define inputs
        self.static_categoricals = static_categoricals
        self.time_varying_known_reals = time_varying_known_reals
        self.time_varying_unknown_reals = time_varying_unknown_reals

        # Create TimeSeriesDataSet objects
        self.training_dataset = TimeSeriesDataSet(
            train_data,
            time_idx="time_idx",
            target=self.target,
            group_ids=["group_id"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=self.static_categoricals,
            time_varying_known_reals=self.time_varying_known_reals,
            time_varying_unknown_reals=self.time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=["group_id"]),
        )
        self.validation_dataset = TimeSeriesDataSet.from_dataset(self.training_dataset, val_data)
        self.test_dataset = TimeSeriesDataSet.from_dataset(self.training_dataset, test_data)

        # Create dataloaders
        self.train_dataloader = self.training_dataset.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=0
        )
        self.val_dataloader = self.validation_dataset.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=0
        )
        self.test_dataloader = self.test_dataset.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=0
        )

    def train_model(self, learning_rate=0.03, hidden_size=16, attention_head_size=4, dropout=0.1, hidden_continuous_size=8, max_epochs=10):
        # Define the Temporal Fusion Transformer model
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            loss=QuantileLoss(), 
        )

        # Determine the accelerator type (TPU, GPU, or CPU)
        accelerator = (
            "tpu" if "COLAB_TPU_ADDR" in os.environ else
            "gpu" if torch.cuda.is_available() else
            "cpu"
        )

        # Add TensorBoardLogger
        logger = TensorBoardLogger("lightning_logs", name="TFT")

        # Define the PyTorch Lightning Trainer
        self.trainer = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            max_epochs=max_epochs,
            gradient_clip_val=0.1,
            log_every_n_steps=1,
            logger=logger,
        )

        # Train the model
        self.trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)

    def test_model(self):
        if self.model is None:
            raise ValueError("Model not trained yet. Please call train_model first.")

        # Make predictions on the test set
        predictions = self.model.predict(self.test_dataloader)

        # Ensure predictions are converted to a NumPy array
        predictions = predictions.cpu().numpy()

        # Get the actuals (true values) from the test dataloader
        actuals = torch.cat([y[0] for x, y in iter(self.test_dataloader)]).cpu().numpy()

        # Flatten predictions and actuals
        predictions = predictions.flatten()
        actuals = actuals.flatten()

        # Calculate MSE
        mse = mean_squared_error(actuals, predictions)
        print(f"Mean Squared Error (MSE) on the test set: {mse}")
        return mse

    def plot_predictions(self, y_axis_label, plot_title):
        if self.model is None:
            raise ValueError("Model not trained yet. Please call train_model first.")

        # Make predictions on the test set
        predictions = self.model.predict(self.test_dataloader, mode="prediction").cpu().numpy().flatten()

        # Get actuals (true values) from the test dataloader
        actuals = torch.cat([y[0] for x, y in iter(self.test_dataloader)]).cpu().numpy().flatten()

        # Adjust predictions to match the test set size
        predictions = predictions[:len(self.test_dates)]  # Ensure predictions match test_dates length
        actuals = actuals[:len(self.test_dates)]          # Ensure actuals match test_dates length
        test_dates = self.test_dates[:len(predictions)]   # Match test dates length

        # Plot predictions vs. actuals
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, actuals, label="Actuals", color="blue")
        plt.plot(test_dates, predictions, label="Predictions", color="orange", linewidth=1)
        plt.title(f"Predicted vs Actual {plot_title}")
        plt.xlabel("Date")
        plt.ylabel(y_axis_label)
        plt.xticks(test_dates[::50], rotation=45)  # Show every 50th date for readability
        plt.gcf().autofmt_xdate()                 # Automatically format x-axis dates
        plt.legend()
        plt.grid()
        plt.show()