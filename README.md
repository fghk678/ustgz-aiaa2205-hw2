# AIAA2205 HW2 - Image Classification with Domain Adaptation

This README provides a step-by-step guide for the AIAA2205 HW2 assignment: **Image Classification with Domain Adaptation using Pretrained Vision Transformers**.

## 1. Project Structure

```
├── data/
│   └── image_list/
│       ├── source.txt
│       ├── target.txt
│       ├── target_eval.txt
│       └── target_to_predict.txt
│   └── source/
│   └── target/
├── models/
│   ├── __init__.py
│   ├── functions.py
│   └── vit.py
├── best_model_baseline.pth
├── dataset.py
├── predict.py
├── prediction_example.csv
├── requirements.txt
├── starter.py
├── train_dann.py
├── utils.py
└── README.md
```

## 2. Environment Setup

We recommend using **conda** for environment management.

```bash
# Create a new environment
conda create -n hw2 python=3.9
conda activate hw2

# Install requirements
pip install -r requirements.txt
# Or, if using conda, you can install pytorch/torchvision according to your CUDA version
```

For more details, refer to `install_conda+pytorch+vscode.md`.

## 3. Data Preparation

1. **Download** the data files from the course/Kaggle competition page.
2. **Extract** `source.tgz` and `target.tgz` into the `data/` directory:

   ```bash
   tar -xzvf source.tgz -C data/
   tar -xzvf target.tgz -C data/
   ```
3. Make sure the image list files are in `data/image_list/`.

## 4. Quick Start

### 4.0. Making Benchmark Predictions

You can generate a benchmark prediction file using the pretrained baseline model (download from [Google Drive](https://drive.google.com/file/d/17gKFTaFUTllnCAEuCVDeHupfi3_VXlCp/view?usp=sharing)):

```bash
python predict.py
```

* This will create a prediction file (e.g., `prediction.csv`) in the required format for Kaggle submission.
* You can compare your future results with this benchmark.

### 4.1. Get Familiar with the Training Pipeline (Task 1)

* Read and understand the provided `starter.py` code.
* Complete all `# TODO` sections in `starter.py`.
* Run the training and validation pipeline:

  ```bash
  python starter.py
  ```
* (Optional) Try visualizing training logs or checking predictions.

### 4.2. Domain Adaptation Training (Task 2)

* Train and experiment with the DANN (Domain-Adversarial Neural Network) baseline:

  ```bash
  python train_dann.py
  ```
* Feel free to modify code, tune hyperparameters, apply data augmentation, or change the model architecture to improve results.

## 5. Making Predictions

To generate predictions for the target domain (for Kaggle submission):

```bash
python predict.py
```

* This will create a prediction file (e.g., `prediction.csv`) in the required format.
* You can refer to `prediction_example.csv` for the expected format.

## 6. Experimentation and Improvement

* Try modifying the code in `models/`, adjusting hyperparameters, or using data augmentation to improve target domain accuracy.
* Document the changes you make for your report.

## 7. Kaggle Submission

1. Go to the course [Kaggle competition page](https://www.kaggle.com/competitions/hkustgz-aiaa-2205-hw-2-2025-summer/leaderboard).
2. Submit your prediction file and check your score.
3. Take a screenshot of both your **public and private leaderboard** result for your report.

## 8. Report

Prepare a brief report (PDF or Markdown) that includes:

* The modifications/experiments you tried.
* How each change affected performance.
* Your main findings and insights.
* A screenshot of your Kaggle leaderboard results.

## 9. Submission Checklist

* [ ] Completed code (including `starter.py`, any modified scripts, etc.)
* [ ] Experiment report (PDF or Markdown)
* [ ] Kaggle leaderboard screenshot (showing both public & private scores)
* [ ] (Optional) Pretrained/best model files if needed for reproduction

**Zip all contents and submit on Canvas.**
