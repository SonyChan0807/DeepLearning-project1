# Prediction of finger movements from EEG recordings via deep neural networks

We experiment with deep neural networks for the prediction of finger movements from EEG recordings. Details can be found in the attached report.

# Contents
------------
  * [Requirements](#requirements)
  * [Brief Project Structure](#brief-project-structure)
  * [Usage](#usage)
  * [Results](#results)
    * [Model Accuracies](#model-accuracies)
    * [Loss Plots](#loss-plots)

# Requirements
------------
## For Windows

  * Windows 10
  * [Anaconda](https://www.anaconda.com/download/) with Python 3
  * [PyTorch](https://anaconda.org/peterjc123/pytorch) 0.3.1
  
## For Linux

# Brief Project Structure
------------

    ├── models                         : Directory containing the scripts (.ipynb) to generate all other models presented in the report
    ├── README.md                      : The README guideline and explanation for our project.
    ├── Report.pdf                     : Report
    ├── run.py                         : Main script to reproduce our best model

# Usage
------------

#### Best Model
To run our best model, simply `$ git clone` the repository then run `$ python run.py` 

#### Other Models
The other models in `.ipynb` are stored in the `models` folder. 

# Results
------------
## Model Accuracies

| **Method**  | **Train** | **Val** | **Test** |
| ------------- | ------------- | ------------- | ------------- |
| SVM | |   | |
| Naive Bayes | | | |
| 3 Layered MLP | | | |

| **Architecture**  | **Train** | **Val** | **Test** |
| ------------- | ------------- | ------------- | ------------- |
| CNN (28 x 28 x 3), FC (25,2) | 90 ± 1% | | |
| CNN (28 x 28 x 3, 28 x 28 x 3), FC (25,25,2) | | |  |

| **Architecture**  | **Train** | **Val** | **Test** |
| ------------- | ------------- | ------------- | ------------- |
| LSTM (25), FC (25,2) |  |  |  |
| LSTM (25), FC (25,25,2) |  | | |

## Loss Plots

