# CYP2C9 Inhibitor Prediction - Machine Learning Pipeline

A end-to-end cheminformatics and machine learning pipeline for predicting **CYP2C9 enzyme inhibitors** using bioactivity data from PubChem. This project covers everything from raw data curation to model validation, walking through each decision made along the way.

---

## Background

**CYP2C9** (Cytochrome P450 2C9) is one of the most important drug-metabolizing enzymes in the human liver. It's responsible for metabolizing roughly 15% of clinically used drugs - including warfarin, ibuprofen, and many anti-diabetics. When a compound inhibits CYP2C9, it can dramatically alter the pharmacokinetics of co-administered drugs, leading to dangerous drug-drug interactions.

Predicting CYP2C9 inhibition computationally before a compound reaches the lab is therefore valuable both for drug safety screening and early-stage drug discovery. This project builds a binary classifier to distinguish **Active (Strong/Moderate)** inhibitors from **Non-inhibitors**, using molecular fingerprints and Lipinski descriptors as features.

**Data source:** PubChem Bioassay [AID: 777](https://pubchem.ncbi.nlm.nih.gov/bioassay/777) - a high-throughput screen measuring CYP2C9 inhibition (% inhibition at 5 µM).

---

## Project Structure

```
CYP2C9_ML/
├── AID_777_datatable.csv              # Raw PubChem assay data
├── bioactivity_preprocessed_data.csv  # Cleaned and filtered dataset
├── df_lipinski.csv                    # Final ML-ready dataset with Lipinski descriptors
├── smiles_chembl.smi                  # SMILES file for PaDEL fingerprinting
├── pubchem_fingerprints.csv           # PubChem fingerprint descriptors
├── Substructure_fingerprints.csv      # Substructure fingerprint descriptors
├── mannwhitney_summary.csv            # Statistical analysis results
├── EDA_results.zip                    # All EDA plots and summary CSVs
└── CYP2C9_ML.ipynb                    # Main notebook (this pipeline)
```

---

## Pipeline Overview

### 1. Data Curation & Preprocessing

The raw PubChem assay file (`AID_777_datatable.csv`) contains metadata rows mixed in with compound records, so the first step is cleaning. Rows with missing `PUBCHEM_CID` values are dropped, and the `% Inhibition at 5 µM` column is coerced to numeric - handling non-numeric entries like `"NONE"` by converting them to `NaN` and dropping them.

After cleaning, the dataset is restructured into four key columns:

| Column | Description |
|---|---|
| `PUBCHEM_CID` | PubChem compound identifier |
| `SMILES` | Canonical SMILES string |
| `Activity_Outcome` | Original PubChem active/inactive label |
| `Inhibition_at_5_uM` | Continuous % inhibition value |

Invalid SMILES strings (null, `"none"`, or empty) are removed before saving the preprocessed dataset.

---

### 2. Exploratory Data Analysis

The EDA phase is more than just plotting - it surfaces and directly addresses two key problems with the raw data before any ML is attempted.

**Duplicate SMILES removal:** After an initial check, duplicate SMILES strings are identified and dropped. For compounds with the same SMILES that appear multiple times (e.g. from different assay runs), inhibition values are aggregated using the **median** to produce a single, more robust measurement per unique structure.

**Addressing class imbalance:** An initial pie chart of the `Activity_Outcome` labels reveals a heavy imbalance between active and inactive compounds - a common issue in bioactivity datasets. Rather than using the raw binary labels, the continuous `% Inhibition at 5 µM` is used to define a three-class system:

| Class | Threshold |
|---|---|
| `Active (Strong/Moderate)` | ≥ 75% inhibition |
| `Weak/Inactive` | 25% – 74% inhibition |
| `Non-inhibitor` | < 25% inhibition |

**Random Under-sampling** (`imblearn.RandomUnderSampler`) is then applied to balance the three classes. Following this, the `Weak/Inactive` class is removed entirely to produce a clean **binary classification** problem between `Active (Strong/Moderate)` and `Non-inhibitor`. This simplification improves both pipeline efficiency and model interpretability.

**Lipinski's Rule of Five descriptors** are calculated for all compounds using RDKit:

- Molecular Weight (MW)
- LogP (lipophilicity)
- Number of Hydrogen Bond Donors
- Number of Hydrogen Bond Acceptors

**Statistical analysis (Mann-Whitney U test)** is run on each descriptor to test whether active and non-inhibitor compounds are drawn from statistically different distributions. Results are compiled into a summary CSV. All boxplots and scatter plots (MW vs LogP coloured by activity class) are exported as PDFs.

---

### 3. Descriptor Calculation & Feature Engineering

Molecular fingerprints are generated using **PaDEL-Descriptor** (via the `padelpy` wrapper). SMILES strings are exported to `.smi` format and two fingerprint types are calculated:

- **PubChem fingerprints** (881-bit, substructure keys)
- **Substructure fingerprints** (from the SubstructureFingerprinter XML)

The fingerprint matrix is merged with the compound metadata and activity labels to form the final feature matrix.

Before modelling, a **Variance Threshold** filter (`threshold = 0.16`, i.e. `p*(1-p)` for `p = 0.8`) is applied to remove near-constant binary features that carry no discriminative signal. This step significantly reduces the feature space.

The final dataset is split **80/20** into training and test sets (`random_state=123`), and all features are standardized using `StandardScaler` fit on the training data only.

---

### 4. Model Training & Evaluation

Five classifiers are trained and evaluated:

| Model | Notes |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Support Vector Machine | RBF kernel |
| K-Nearest Neighbours | k=5 |
| Random Forest | 100 estimators |
| Naive Bayes | Gaussian |

Each model is evaluated on four metrics: **Accuracy, Precision, Recall, and F1 Score**, all computed with `pos_label='Active (Strong/Moderate)'`. A grouped bar chart compares all five models side by side.

A **confusion matrix** is plotted for Logistic Regression, breaking down true positives, true negatives, false positives (wasted lab resources), and false negatives (missed drug candidates).

**5-Fold Stratified Cross-Validation** is run on Logistic Regression using a custom F1 scorer, checking for consistency across folds. Low standard deviation across folds confirms that the model generalises rather than overfitting to one particular train/test split.

**Y-Randomization** is performed on Random Forest over 5 iterations: the target labels are shuffled while keeping features intact. The shuffled-label accuracy (~0.506) is substantially lower than the real model accuracy (~0.687), confirming that the model is learning genuine structure-activity relationships and not just a statistical artefact.

---

### 5. Feature Importance

The top 10 most important descriptors from the Random Forest model are extracted and visualised as a horizontal bar chart. This step gives chemical insight into which molecular features drive CYP2C9 inhibition predictions - useful for understanding the model and informing future compound design.

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
rdkit
padelpy
lazypredict
shap
scipy
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn rdkit padelpy lazypredict shap scipy
```

> **Note:** This notebook was developed in Google Colab. Google Drive is used for file persistence across sessions. If running locally, update all file paths accordingly.

---

## How to Run

1. Download the raw assay data from [PubChem AID 777](https://pubchem.ncbi.nlm.nih.gov/bioassay/777) and save it as `AID_777_datatable.csv`.
2. Open `CYP2C9_ML.ipynb` in Google Colab or Jupyter.
3. Run cells in order - each section depends on the outputs of the previous one.
4. PaDEL XML descriptor files are pulled automatically from the course GitHub repo during the fingerprint calculation step.

---

## What's Next

The next phase of this project will:

- Save the best-performing model to disk
- Test the model on **known CYP2C9 inhibitors** from literature
- Screen **large libraries of natural compounds** to identify novel CYP2C9 inhibitor candidates

---

## References

- PubChem BioAssay AID 777: CYP2C9 inhibition screen
- Lipinski CA et al. *Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings.* Adv Drug Deliv Rev. 2001.
- Yap CW. *PaDEL-descriptor: An open source software to calculate molecular descriptors and fingerprints.* J Comput Chem. 2011.
- Imbalanced-learn: Lemaître G et al. *Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning.* JMLR. 2017.

---

> This project is in progress, External Validation against known inhibitors and non-inhibtors, large scale screening will be done against natural product databases. Top targets will be further validated using molecular docking technique. 
