# Transformer for Public Safety

## Overview

The Transformer for Public Safety project aims to develop a predictive model using transformer architecture to forecast emergency events, such as medical and fire, in large cities. By leveraging historical data and advanced machine learning techniques, the project seeks to enhance emergency response capabilities and improve public safety.

Emergency response agencies need accurate predictions to allocate resources effectively. Existing regression models lack the ability to fully capture spatial and temporal dependencies in the data, leading to suboptimal predictions.

Our transformer model predicts the number of events that will occur in a neighborhood over a weekly period. It utilizes various features, including location, time, building types, and historical event patterns, to make these predictions.

## Running the project

### 1. Preparing Python Environment

``` bash
pip install -r requirements.txt
```

### 2. Pre-processing Dataset

``` bash
python model/data_processing_efrs.py
```

### 3. Train and Validation

``` bash
python model/train_evaluate.py
```
