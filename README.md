# Breast cancer classification with an MLP

The main model is an MLP; logistic regression is a comparison baseline. We reserve a stratified 20% test set and perform five-fold cross-validation only within the 80% development set. Architecture, probability thresholds, and MLP training duration are selected before final test evaluation.

- `Assignment_Report.docx`: Word document for submission.
- `REPORT.md`: editable source of the report, with explanations, code snippets, and results.
- `breast_cancer_mlp.ipynb`: self-contained notebook; run all cells locally or in Colab.
- `experiment.py`: equivalent Python experiment.
- `data/`: original UCI dataset and feature documentation.
- `results/`: generated development metrics, predictions, test metrics, and plots.

## Run locally

```bash
python -m pip install -r requirements.txt
python experiment.py
```

Use the project folder as the working directory. Local data is included. The notebook installs `ucimlrepo` for its download fallback; if using the script without local data, install `ucimlrepo` too.

The final metrics are in `results/final_metrics.json`. `out_of_fold_predictions.csv` contains development predictions only; `test_predictions.csv` contains final held-out predictions. Rerunning overwrites corresponding result files. Different TensorFlow versions/hardware can produce numerical differences.

The full dataset was used in earlier exploratory work; this revised protocol separates development and test use within the new run. It is an educational experiment, with no external validation.

Dataset: [UCI Wisconsin Diagnostic Breast Cancer](https://doi.org/10.24432/C5DW2B).
