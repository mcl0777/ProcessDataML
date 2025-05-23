# Machine Learning for Industrial Quality Prediction (Master’s Thesis)

This repository contains the full Machine Learning pipeline developed as part of a master's thesis on the prediction and classification of quality-related outcomes in industrial manufacturing. 

All scripts and notebooks are written in **German**, but this README is in **English** to facilitate sharing and evaluation. German filenames and keywords are shown in parentheses.

---

## Project Structure

```
.
├── data_preparation/         # Data preprocessing and feature construction (Datenvorbereitung)
├── datasets/                 # Raw and processed datasets (Datensätze)
├── helpers/                  # Utility functions (Hilfsfunktionen)
├── results/                  # Visualizations and model results (Ergebnisse)
├── visualizations/           # Charts and explanatory graphics (Visualisierungen)
├── Dockerfile                # Docker environment specification
├── requirements.txt          # Package dependencies
                              # Analysis notebooks (z. ​z. B. `Prozessqualität.ipynb`, `Einbauprognose.ipynb`)
```

---

## Target Variables

This project models and evaluates the following six prediction targets:

1. **Process quality** (`Prozessqualität`) – classification
2. **Sample position** (`Probenposition`) – classification
3. **Sample height** (`Probenhoehe`) – regression
4. **Material prediction** (`Materialprognose`) – classification
5. **Installation prediction** (`Einbauprognose`) – classification
6. **Component temperature** (`Bauteiltemperatur`) – regression

---

## ML Models

Classification:

* k-Nearest Neighbors (`kNN`)
* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* AdaBoost
* XGBoost

Regression:

* XGBoost (main model)
* SVR, Random Forest Regressor (comparisons)

---

## Feature Engineering

The feature engineering is specifically tailored to a thermo-mechanical forming process (Warmumformung) and includes:

Time-series aggregates and statistics from sensor data (e.g. average, max, time to peak)

Temperature- and force-based differential features (e.g. Δ temperature, Δ force over forming time)

Segment-wise window operations (e.g. rolling averages during heating/pressing)

Process-phase indicators (e.g. „heating“, „positioning“, „pressing“ phases)

Manual engineering of features linked to material ID, location and equipment conditions

Feature generation is handled by feature_engineering.py, and fully integrated via process_data.py.

Three main feature groups are created and evaluated:

aggregated_features (aggregierte Merkmale)

new_features (neue konstruktive Merkmale, speziell für Warmumformung)

process_features (prozessspezifische Merkmale, z. B. Bauteiltemperatur, Motordrehzahl)

---

## Feature Selection & Reduction

For each target variable, features were reduced using a combination of:

SHAP analysis (SHapley Additive Explanations)

Correlation filtering (Spearman/Pearson)

Variance Inflation Factor (VIF) analysis

Each selected feature set was benchmarked across multiple models and compared against full and reduced variants. The goal was to achieve optimal performance with minimal redundancy and high interpretability.

All selection results are documented in best_models/ (e.g. selected_features_Prozessqualität.csv, vif_plot_*, SHAP summary plots).

---

## Run with Docker

No need to set up a local Python environment.

### 1. Build the Docker image

```bash
docker build -t process-data-ml .
```

### 2. Start the container (Jupyter notebook interface)

```bash
docker run -p 8888:8888 process-data-ml
```

Then open the browser link shown in the terminal (e.g. `http://127.0.0.1:8888/...`)

> The container launches Jupyter by default. To run a specific script instead, modify the `CMD` line in the `Dockerfile`.

---

## Running Locally (without Docker)

### Python Environment

* Python 3.10+
* Pip or Conda

### Installation

```bash
git clone https://github.com/mcl0777/ProcessDataML.git
cd ProcessDataML

# with venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# OR with conda
conda create -n ml_env python=3.12
conda activate ml_env
pip install -r requirements.txt
```

---

## Notebooks (Analyses)

Launch Jupyter and open one of the following notebooks:

```bash
jupyter notebook
```

* `Prozessqualität.ipynb` – Process quality classification
* `Probenposition.ipynb` – Sample position classification
* `Probenhoehe.ipynb` – Sample height regression
* `Materialprognose.ipynb` – Material type classification
* `Einbauprognose.ipynb` – Installation prediction
* `Bauteiltemperatur.ipynb` – Component temperature regression

---

## Results

* Model metrics and plots are saved in `results/`
* Final models and summaries in `best_models/`
* Visual explanations using SHAP available per model

---

## Limitations

* Large datasets (e.g. `.pkl` > 100 MB) are excluded from GitHub via `.gitignore`
* Models are not deployed for production but built for academic analysis
* Use your own data or synthetic examples to test notebooks

---

## License

This repository is intended for demonstration and educational use.
Contact the author for further details if reuse is intended.

---

## Contact

**Author:** Moritz Lepper
**Project:** Master's thesis in Industrial Data Science / Engineering
**Use Case:** ML-based quality assurance in manufacturing
