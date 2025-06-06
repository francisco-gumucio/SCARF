# SCARF: Algorithmic Recourse in Sequential Decision-Making for Long-Term Fairness

**Submission Code for Review**

This repository contains the complete implementation and experimental code for reproducing all results presented in our paper submission.

## Quick Reproduction

To reproduce all paper results:

```bash
# Download repository from anonymous link
wget -O SCARF.zip https://anonymous.4open.science/r/SCARF-E353/archive.zip
unzip SCARF.zip
cd SCARF-E353
pip install -r requirements.txt
jupyter notebook
# Run both notebooks: toy_data_gen.ipynb.ipynb and taiwan_data_gen.ipynb.ipynb
```

## Environment Setup

### Requirements
- Python 3.8+
- Jupyter Notebook/Lab
- CUDA 11.0+ (for GPU experiments)
- 16GB RAM minimum

### Installation
```bash
# Create environment
conda create -n scarf python=3.8
conda activate scarf

# Install dependencies
pip install -r requirements.txt

# Install Jupyter (if not already installed)
pip install jupyter

# Start Jupyter
jupyter notebook
```

## Results Reproduction

### Table 1: Results on Synthetic Dataset

| Method               | Accuracy | Long-Term Fairness | Short-Term Fairness | Runtime    |
|----------------------|----------|--------------------|----------------------|------------|
| SCARF (Ours)         | 0.86095  | 0.06273            | 0.0714               | ~1hr       |
| Equal Improvability  | 0.83999  | 0.06719            | 0.36873              | ~0.75hrs   |


**Reproduce with:**
```bash
# Open and run the Dataset 1 notebook
jupyter notebook notebooks/toy_data_gen.ipynb
```

### Table 2: Results on Taiwan

| Method | Accuracy | Long-Term Fairness | Short-Term Fairness | Runtime |
|--------|----------|-------------------|--------------------|---------| 
| SCARF (Ours) | 0.86095 | 0.06273 | 0.0714 | ~1hr |
| Equal Improvability | 0.83999 | 0.06718595 | 0.36873019 | ~0.75hrs |

**Reproduce with:**
```bash
# Open and run the Dataset 2 notebook
jupyter notebook notebooks/taiwan_data_gen.ipynb
```

### Table 3: Ablation Study (Both Datasets)

| Component  | Synthetic Dataset LT Fairness | Synthetic Dataset ST Fairness | Taiwan Dataset LT Fairness | Taiwan Dataset ST Fairness |
|------------|-------------------------------|-------------------------------|----------------------------|----------------------------|
| Full SCARF | 0.06272                        | 0.0714                         | 0.03631                    | 0.06159                    |
| w/o LSTM   | 0.11600                        | 0.34627                        | 0.05166                    | 0.19833                    |


**Reproduce with:**
```bash
# Ablation results are generated within each dataset notebook
# Run both notebooks to get complete ablation analysis
jupyter notebook notebooks/toy_data_gen.ipynb  # Generates Dataset 1 ablation
jupyter notebook notebooks/taiwan_data_gen.ipynb  # Generates Dataset 2 ablation
```

### Table 4: Computational Efficiency

| Method | Synthetic Dataset Time | Dataset 2 Time | Memory (GB) |
|--------|----------------|----------------|-------------|
| SCARF | ~4hrs | ~5hrs | 1GB |
| Baseline-1 | ~3hrs | ~4hrs | 1GB |

**Reproduce with:**
```bash
# Efficiency metrics are computed within each notebook
# Both notebooks include timing and memory profiling
```

## Notebook Structure

### `notebooks/dataset1_experiments.ipynb`
This notebook contains the complete experimental pipeline for Dataset 1:

**Sections:**
1. **Data Loading & Preprocessing** - Load and prepare Dataset 1
2. **Model Training** - Train SCARF and all baseline methods
3. **Evaluation** - Test all methods and compute metrics
4. **Ablation Study** - Test SCARF without key components
5. **Efficiency Analysis** - Runtime and memory profiling
6. **Results Export** - Save results to CSV files
7. **Visualization** - Generate plots and figures

### `notebooks/dataset2_experiments.ipynb`
This notebook contains the complete experimental pipeline for Dataset 2:

**Sections:**
1. **Data Loading & Preprocessing** - Load and prepare Dataset 2
2. **Model Training** - Train SCARF and all baseline methods
3. **Evaluation** - Test all methods and compute metrics
4. **Ablation Study** - Test SCARF without key components
5. **Efficiency Analysis** - Runtime and memory profiling
6. **Results Export** - Save results to CSV files
7. **Visualization** - Generate plots and figures

## Data

The datasets used in this paper are included in the repository under the `data/` directory.

### Data Sources
- Synthetic Dataset: Created according to the method shown in the notebook.
- Taiwan Dataset: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

## Running Experiments

### Complete Reproduction
To reproduce all results from both datasets:

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Run Synthetic Dataset Experiments:**
   - Open `notebooks/toy_data_gen.ipynb`
   - Run all cells: `Cell → Run All`

3. **Run Dataset 2 Experiments:**
   - Open `notebooks/taiwan_data_gen.ipynb`
   - Run all cells: `Cell → Run All`

### Command Line Execution
```bash
# Execute notebooks from command line
jupyter nbconvert --to notebook --execute notebooks/toy_data_gen.ipynb
jupyter nbconvert --to notebook --execute notebooks/taiwan_data_gen.ipynb
```

### Partial Execution
Each notebook is structured with clear sections. You can run individual sections by:
1. Running cells up to desired section
2. Using notebook section headers to navigate
3. Restarting kernel between different method comparisons if needed

## Expected Runtime

**Runtime includes complete pipeline: training + testing + evaluation for all methods**

| Notebook | Dataset Size | Runtime Breakdown | Total Runtime |
|----------|--------------|-------------------|---------------|
| `toy_data_gen.ipynb` | 700 samples for training, 125 samples for validation, 200 samples tested over 30 seeds | Training: ~2-3 hrs<br>Testing: ~30 min<br>Evaluation: ~30 min | **~3-4 hours** |
| `taiwan_data_gen.ipynb` | 700 samples for training, 125 samples for validation, 200 samples tested over 30 seeds | Training: ~2-3 hrs<br>Testing: ~30 min<br>Evaluation: ~30 min | **~3-4 hours** |
| **Total** | 1,400 samples for training, 250 samples for validation, 400 samples tested over 30 seeds | **Complete reproduction** | **~6-8 hours** |

**What's included in runtime:**
- **Training**: SCARF model + all baseline methods + ablation variants
- **Testing**: Inference on test sets for all trained models
- **Evaluation**: Metric computation, statistical tests, efficiency analysis
- **Visualization**: Figure generation and results export

## Hardware Requirements

- **Minimum**: 16GB RAM, 4-core CPU
- **Recommended**: 32GB RAM, 8-core CPU, GPU with 8GB VRAM
- **Storage**: 20GB free space for intermediate results and model checkpoints

## Repository Structure

```
SCARF/
├── SimLoan/
      VACA/                         # Core implementation
         data_modules/
         datasets/
         models/
         modules/
         utils/
         baseline.py
         taiwan_data_gen.ipynb
         toy_data_gen.ipynb
         vaca_model.py
         weights.yaml
      data/
         default of credit card clients.xls
      res/
      src/                              
└── README.md                         # This file
```

## Verification

### Result Verification
Each notebook generates:
- **Numpy Files** with exact numerical results
- **Figures** matching those in the paper

### Cross-Validation
Both notebooks include:
- Multiple random seeds for statistical reliability
- Cross-validation splits where applicable

## Troubleshooting

### Memory Issues
```bash
# If notebooks run out of memory:
# 1. Restart kernel between major sections
# 2. Reduce batch size in configuration cells
# 3. Use GPU if available to free CPU memory
```

### Long Runtime
```bash
# To reduce runtime for testing:
# 1. Modify epoch numbers in early cells
# 2. Run only specific methods by commenting out baseline sections
```

### Jupyter Issues
```bash
# If Jupyter becomes unresponsive:
jupyter notebook stop
jupyter notebook clean
jupyter notebook
```

## Notes for Reviewers

- **Two Complete Pipelines**: Each dataset has its own comprehensive notebook
- **Self-Contained**: Each notebook loads data, trains models, and generates all results
- **Modular Sections**: Clear section headers allow running specific parts
- **All Results**: Both notebooks together reproduce all paper tables and figures
- **Intermediate Outputs**: Results saved throughout execution for verification

## Contact

For reproduction issues or questions about this anonymous submission, please refer to the paper submission system or contact through the conference review process.

**Anonymous Repository**: https://anonymous.4open.science/r/SCARF-E353/

---

**Reviewer Instructions**: 
1. Install requirements: `pip install -r requirements.txt`
2. Start Jupyter: `jupyter notebook`
3. Run `dataset1_experiments.ipynb` and `dataset2_experiments.ipynb`
4. Check `results/` directory for generated tables and figures
5. Total runtime: ~6-8 hours for complete reproduction
