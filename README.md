# Time Series Forecasting with Temporal Fusion Transformers
Make accurate forecasts using Temporal Fusion Transformers with PyTorch.

## Overview
This repository implements the PyTorch Forecasting Temporal Fusion Transformer (TFT) for interpretable multi-horizon time series forecasting. The provided pipeline automatically prepares the dataloaders, trains the model, and plots the performance of the model on the test set.

## Example Forecasts
Below are examples of the model's forecasting performance on various datasets:

![Forecast Example 1](/plots/yield_curve_prediction.png)
![Forecast Example 2](/plots/web_traffic_prediction.png) 
![Forecast Example 3](/plots/weather_prediction.png)

## Prerequisites
It is recommended to make the data stationary before training the model. A conda environment is provided in the `environments` directory to make it easier to run the code.

## How to Use the Provided Conda Environment
This repository provides the conda environment which was used throughout the development of this project. It is known to be compatible with Nvidia CUDA version 12.4 and with TPUs available on Google Cloud.

To install the conda environment, run the following command:
```
conda env create -f environments/environment.yml --name tft
```

To activate the conda environment, run the following command:
```
conda activate tft
```

## Inputs
The following table explains the three different categories of input features that are used by the TFT model to ensure proper configuration:

| Feature Type              | Description                                                                 |
| ------------------------- | --------------------------------------------------------------------------- |
| Static Categoricals       | Categorical features that are constant for all time steps (note that this is only relevant if the dataset contains multiple groups; see `group_id`). |
| Time-Varying Known Reals  | Continuous features that vary over time but are known in advance for all time steps (e.g., weather forecasts, holiday indicators). |
| Time-Varying Unknown Reals| Continuous features that vary over time and are unknown for future time steps (e.g., sales, stock prices). |

### Denoting the Group ID
The `group_id` argument is used to denote the group that the data belongs to. If the dataset contains multiple groups, then this argument must be provided. If the dataset contains a single group, then this argument will be set to `default_group`. The provided examples only use a single group, so this argument is not used.

## How to Launch Tensorboard for Visualizing Training/Validation Losses
Run the following command to start the tensorboard server:
```
tensorboard --logdir lightning_logs
```

## Background Information
The TFT model combines elements of transformers, LSTMs, GRUs, and Resnets to create a model that is highly effective at forecasting tasks. A key differentiator of the TFT as compared to the LSTM is the multi-head attention mechanism,
which allows the model to capture long range dependencies in the data. While LSTM processes the input sequence purely sequentially, the TFT's attention mechanism processes the entire input sequence at once in parallel. This allows the TFT
to recognize relationships between elements of the input sequence that are temporally distant from each other.  
  
This project provides a working pipeline of the TFT model provided by the PyTorch Forecasting library. The pipeline is a good starting point, but it can (and probably should) be adapted using other modules from this library to make it more specialized to the desired forecasting task.  

### Model Architecture
The below table describes the various modules of the TFT and their purpose:

| Name                             | Type                                | Description                                                                                     |
|----------------------------------|-------------------------------------|-------------------------------------------------------------------------------------------------|
| loss                             | QuantileLoss                        | The loss function used for training the model to minimize prediction errors.                   |
| logging_metrics                  | ModuleList                          | A module to track and log evaluation metrics during training and testing.                      |
| input_embeddings                 | MultiEmbedding                      | Handles embeddings for categorical features or other inputs into the model.                    |
| prescalers                       | ModuleDict                          | Applies scaling to input features for normalization.                                           |
| static_variable_selection        | VariableSelectionNetwork            | Selects relevant static features to include in the model's computation.                       |
| encoder_variable_selection       | VariableSelectionNetwork            | Selects important variables for the encoder's input sequence.                                  |
| decoder_variable_selection       | VariableSelectionNetwork            | Selects relevant variables for the decoder's input during prediction.                         |
| static_context_variable_selection| GatedResidualNetwork                | Processes static context information for variable selection.                                   |
| static_context_initial_hidden_lstm| GatedResidualNetwork               | Initializes the hidden state of the LSTM using static context features.                       |
| static_context_initial_cell_lstm | GatedResidualNetwork                | Initializes the cell state of the LSTM using static context features.                         |
| static_context_enrichment        | GatedResidualNetwork                | Enriches static context information for further processing.                                    |
| lstm_encoder                     | LSTM                                | Processes the input sequence to encode temporal dependencies.                                  |
| lstm_decoder                     | LSTM                                | Decodes the output sequence from the encoded temporal dependencies.                           |
| post_lstm_gate_encoder           | GatedLinearUnit                     | Adds a gating mechanism to filter encoded information after the LSTM.                         |
| post_lstm_add_norm_encoder       | AddNorm                             | Applies addition and normalization to improve stability and performance.                      |
| static_enrichment                | GatedResidualNetwork                | Enriches static feature representation after initial processing.                              |
| multihead_attn                   | InterpretableMultiHeadAttention     | Applies attention to capture dependencies between time steps.                                  |
| post_attn_gate_norm              | GateAddNorm                         | Combines gating, addition, and normalization after attention.                                 |
| pos_wise_ff                      | GatedResidualNetwork                | Applies position-wise feed-forward layers for feature refinement.                             |
| pre_output_gate_norm             | GateAddNorm                         | Prepares processed features for output by applying gating and normalization.                  |
| output_layer                     | Linear                              | Maps the processed features to the target variable's output space.                            |

