## Empirical AutoML on Tabular Data

For this assignment, we compared three approaches of hyperparameter optimization: Random Search, SMBO and Hyperband. 
We compared how well these algorithms performed on a Random Forest Classifier across five different tabular datasets. 
an untuned Random Forest was used as a baseline and TabPFN was evaluated as a pre-trained tabular foundation model on the largest dataset


## Setup

Use Python 3.11 or newer. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Datasets

The experiments use five different OpenML datasets:

- breast-w
- credit-g
- phoneme
- electricity
- covertype

## Running the experiment

The experiments were conducted using the provided course profile. The smoke profile can be used for a quick functionality test.

To test if everything works, you can use the following command in the terminal:

```bash
python experiment.py --dataset breast-w --profile smoke --methods default
```

To run the Random Forest experiments:

```bash
python experiment.py --dataset all --profile course --methods default random smbo hyperband --seeds 17 18 19
```

To run the TabPFN experiments: 

```bash
python experiment.py --dataset covertype  --profile course --methods foundation  --seeds 17 18 19
```

Due to GPU memory limitations, TabPFN was fitted on a stratified subset of 10,000 samples from the combined training and validation data. Evaluation was performed on the complete test set.

## TabPFN authentication token
TabPFN may require an authentication token. Set your own token using the TABPFN_TOKEN environment variable before running the foundation experiment.

On Window Powershell, use:

```powershell
$env:TABPFN_TOKEN = "YOUR_TOKEN"
```
On linux/macOS, use:
```bash
export TABPFN_TOKEN="YOUR_TOKEN"
```
Our token is not included with this submission. 

## Results

The results are automatically saved as JSON files in the results/ directory after each experiment is completed.
The recorded metrics include accuracy, Macro-F1 and runtime. 

For the Random Forest optimization methods, runtime represents hyperparameter search time.
 For TabPFN, runtime includes fitting and inference but excludes pre-training.

## Figure creation
The figures in the report can be made with the terminal command:
```bash
python -m scripts.analysis
```
The scrips reads the experiment results from the results directory and saves the figures in the analysis directory.

## Hardware used for the experiment
CPU: Intel(R) Core(TM) i9-14900HX
GPU: NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM)








