# GridPulse: A Benchmark for Electricity Price Forecasting under Extreme Market Dynamics

## 1. Overview

GridPulse is a challenging benchmark for electricity price forecasting in provincial spot markets. It contains 17,496 hourly observations from two markets with distinct mechanisms, together with 21 temporally aligned covariates and leakage-free preprocessing. Unlike existing public benchmarks that mainly reflect relatively stable market dynamics, GridPulse captures heterogeneous and highly volatile price regimes characterized by frequent negative prices, sharp price spikes, and abrupt regime shifts. It provides a unified evaluation protocol with three bounded, scale-normalized metrics—RSI, QRS, and HARI—and evaluates 11 representative forecasting models under both normal and extreme market conditions. GridPulse also includes PulseDiff as a reproducible baseline, offering a comprehensive testbed for developing robust electricity price forecasting methods in volatile markets.

## 2. Implement

GridPulse requires Python 3.10 or later. Clone the repository and install the required dependencies:

```bash
git clone https://github.com/77-qiqi-wang/GridPulse.git
cd GridPulse

conda create -n gridpulse python=3.10 -y
conda activate gridpulse

pip install -r requirements.txt
```

The training code automatically uses CUDA when a compatible GPU is available; otherwise, it runs on CPU.

Train and evaluate PulseDiff on the dataset:

```bash
# Run on the Liaoning Dataset
python run_liaoning.py

# Run on the Shandong Dataset
python run_shandong.py
```

The results are saved to `outputs_liaoning/` and `outputs_shandong/`, respectively.

## 3. Dataset
