# Federated Learning Models for Privacy-Preserving Healthcare Analytics

**B.Tech Final-Year Capstone Project | Computer Science and Engineering**

A simulation-based federated learning platform that explores collaborative healthcare analytics while keeping raw training data at individual client nodes.

---

## 1. Project Overview

Healthcare organizations maintain valuable patient data that can support machine learning and healthcare analytics. However, sharing raw patient records between institutions can create privacy, security, and regulatory concerns.

**Federated Learning (FL)** provides an alternative approach. Instead of collecting all records in one central location, participating clients train models locally and share model updates with a coordinator. The coordinator aggregates these updates to produce a shared global model.

This project implements and evaluates multiple federated learning approaches using synthetically generated healthcare-style datasets. It includes a Python-based machine learning pipeline and an interactive Streamlit dashboard for exploring experiments and results.

> **Project scope:** This is an academic simulation using synthetic data. It is not a medical diagnostic tool and has not been validated for clinical use.

## 2. Problem Statement

Traditional centralized machine learning requires data from multiple healthcare organizations to be collected in one location. This can increase the risk associated with transferring and storing sensitive patient information.

The project investigates the following question:

**How can multiple healthcare data holders collaboratively train and evaluate machine learning models without directly pooling their raw training records?**

The project explores federated learning methods, model performance, communication overhead, and privacy-related considerations.

## 3. Objectives

- Implement a federated learning workflow with multiple simulated healthcare clients.
- Keep raw training records local to each client during federated training.
- Implement and compare federated learning algorithms.
- Explore horizontal and vertical federated learning approaches.
- Evaluate models using standard classification metrics.
- Analyze communication overhead and training behavior.
- Develop an interactive Streamlit dashboard for experiment visualization.
- Document the system architecture, reproducibility steps, and limitations.

## 4. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical operations and model parameter processing |
| Pandas | Dataset preparation and experiment result handling |
| Scikit-learn | Data preprocessing and evaluation metrics |
| Streamlit | Interactive dashboard |
| Matplotlib / Plotly | Charts and visualizations |
| Firebase Cloud Firestore | Experiment metadata and history, when configured |
| Python unittest | Automated testing |
| Google Colab | Optional cloud-based notebook environment |

## 5. System Architecture

The project simulates multiple healthcare clients connected to a central coordinator.

```text
              Synthetic Healthcare Data
                         |
          +--------------+--------------+
          |              |              |
      Hospital 1     Hospital 2     Hospital 3
          |              |              |
     Local Training Local Training Local Training
          |              |              |
     Model Updates  Model Updates  Model Updates
          |              |              |
          +--------------+--------------+
                         |
                  FL Coordinator
                         |
                 Model Aggregation
                         |
                  Global Model
                         |
                 Model Evaluation
                         |
               Metrics and Results
                         |
                Streamlit Dashboard
                         |
              Experiment Visualization
```

### Federated Training Workflow

1. Initialize a shared model architecture.
2. Distribute the current global model to the simulated clients.
3. Train each client model using its local training data.
4. Collect model parameters or updates from participating clients.
5. Aggregate the updates using the selected federated algorithm.
6. Update the global model.
7. Repeat for the configured communication rounds.
8. Evaluate the resulting model using a held-out test dataset.

Raw training records are not intended to be transferred to the coordinator during the federated training workflow.

## 6. Implemented Federated Learning Approaches

The project explores the following approaches:

### Horizontal Federated Learning (HFL)

Horizontal federated learning is used when participating clients have similar feature structures but different records.

The project uses three simulated healthcare clients with different data distributions.

**FedAvg — Federated Averaging**

FedAvg combines client model parameters using a sample-weighted average.

**FedProx**

FedProx adds a proximal regularization term to the local training objective. It is intended to reduce client drift when client datasets have different statistical distributions.

### Secure Aggregation (SecAgg)

The project includes a simulated pairwise additive masking approach for aggregating model updates.

The purpose is to demonstrate how individual client updates can be masked so that the coordinator can aggregate them without directly receiving the unmasked individual updates.

**Important:** This implementation is an academic simulation. It should not be treated as a production-grade cryptographic protocol or as a guarantee against all privacy attacks.

### Vertical Federated Learning (VFL)

Vertical federated learning is explored when different clients hold different feature groups for aligned records.

The project includes a SplitNN-style simulation in which model computation is divided between participating feature-owning clients.

### Federated Transfer Learning (FTL)

Federated transfer learning explores transferring useful model knowledge between clients with different data distributions and different amounts of available data.

### Hybrid CNN-LSTM

The project includes a longitudinal healthcare-style experiment using synthetic patient sequences.

- The CNN component extracts local temporal patterns.
- The LSTM component models sequential dependencies.
- The experiment compares temporal models and federated training behavior.

## 7. Dataset and Experimental Setup

The project uses **synthetically generated healthcare-style data**.

The data is designed for demonstrating machine learning workflows and federated learning behavior. It does not represent real patient records.

The experiments include:

- Multiple simulated healthcare clients.
- Different client data distributions.
- Tabular healthcare-style features.
- Synthetic longitudinal patient sequences.
- Training, validation, and testing partitions.
- Patient-level partitioning where applicable to the experiment.

Synthetic data is useful for testing and demonstrating the implementation. However, results obtained from synthetic data do not establish clinical validity or real-world healthcare performance.

## 8. Model Evaluation

The project uses standard classification metrics to evaluate model performance.

| Metric | Description |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of predicted positive cases that are correct |
| Recall | Proportion of actual positive cases detected |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Ability of the model to distinguish between classes |
| Loss | Difference between model predictions and target values |
| Confusion Matrix | Breakdown of true and false predictions |

The project also analyzes communication-related measurements, including model update size and communication volume across training rounds.

Reported results should be interpreted in the context of the synthetic datasets, simulation settings, and evaluation methodology used.

## 9. Streamlit Dashboard

The Streamlit application provides an interactive interface for exploring the project.

Depending on the configured modules, dashboard functionality includes:

- Project overview and architecture.
- Federated training controls.
- Training loss and convergence visualization.
- Model performance comparisons.
- Evaluation metrics and confusion matrices.
- Client-level analysis.
- Robustness and communication analysis.
- Longitudinal model visualization.
- Experiment history and metadata.
- Privacy and methodology information.

The dashboard is intended to help users understand the workflow and inspect the results of the experiments.

## 10. Project Structure

The following is a simplified overview of the project structure. Individual files and modules may vary depending on the current project version.

```text
hospital project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── aggregation.py
│   ├── client.py
│   ├── evaluation.py
│   ├── experiment.py
│   ├── secure_aggregation.py
│   └── ...
│
├── data/
│   └── ...
│
├── results/
│   └── metrics/
│
├── tests/
│   └── ...
│
└── ...
```

### Main Components

| Component | Responsibility |
|---|---|
| `app.py` | Streamlit dashboard entry point |
| `src/` | Federated learning algorithms and supporting modules |
| `data/` | Synthetic datasets and experiment inputs |
| `results/` | Generated experiment outputs and metrics |
| `tests/` | Automated unit and integration tests |
| `requirements.txt` | Python dependencies |

## 11. Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version.
- Git, if cloning the repository.
- A terminal or command prompt.
- A Google account, if using Google Colab.

### Clone the Repository

```bash
git clone <your-repository-url>
cd "hospital project"
```

If you already have the project ZIP file, extract it and open a terminal inside the project folder.

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

### Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 12. Running the Project

### Run the Streamlit Dashboard

From the project root directory:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in a browser to access the dashboard.

### Run Federated Training

The project also provides command-line experiment modules.

For example, the standard FedAvg experiment can be run using the project's experiment entry point:

```bash
python src/experiment.py
```

The default settings and supported arguments are defined in the corresponding experiment module.

### Google Colab

Google Colab can be used to execute the project's Python code in a cloud-hosted notebook environment.

A typical workflow is:

1. Open a Colab notebook.
2. Upload the latest project ZIP file.
3. Extract the ZIP file.
4. Set the project directory.
5. Install the dependencies.
6. Run the required training and evaluation commands.
7. Inspect the generated output.

Example extraction:

```python
import zipfile
import os

zip_file = "/content/hospital project.zip"
extract_path = "/content/project"

with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print(os.listdir(extract_path))
```

After extraction, locate the actual project folder and set the working directory accordingly.

**Note:** Colab's temporary runtime storage may be cleared when the runtime disconnects. Save important outputs to Google Drive or download them before ending the session.

## 13. Running Automated Tests

The project uses Python's built-in `unittest` framework.

Run the complete test suite from the project root:

```bash
python -m unittest discover -s tests -v
```

The latest reported test run completed:

```text
Ran 129 tests
```

This test count reflects the reported execution at the time of documentation. Run the test command again after making code changes to verify the current version.

Tests help validate implementation behavior, but passing tests do not by themselves establish clinical validity, production security, or correctness under every possible input.

## 14. Firebase and Experiment Metadata

The project supports Google Cloud Firestore for experiment metadata and history when Firebase is configured.

The intended use is to store information such as:

- Experiment identifiers.
- Training configuration.
- Communication round metadata.
- Evaluation summaries.
- Convergence information.
- Experiment timestamps.

Raw patient records and local client datasets should not be uploaded to Firestore.

Firebase credentials must be stored securely using environment variables or an appropriate secret-management mechanism. Do not commit service account credentials or private keys to GitHub.

## 15. Privacy and Security Considerations

The project demonstrates federated learning concepts, but several distinctions are important.

### What the Simulation Demonstrates

- Local client training.
- Aggregation of model updates.
- Federated training workflows.
- Simulated secure aggregation.
- Separation of raw synthetic client data from the coordinator workflow.
- Metadata-only experiment tracking when configured as intended.

### What It Does Not Guarantee

- Absolute privacy against all inference attacks.
- Formal differential privacy.
- Production-grade cryptographic security.
- Compliance certification under HIPAA, GDPR, or other regulations.
- Security of a real multi-hospital deployment.
- Clinical reliability or diagnostic effectiveness.

**Federated Learning is not the same as Differential Privacy.** Model updates can still contain information about training data. Differential privacy and other security mechanisms may be required for stronger privacy guarantees.

The secure aggregation component in this project is a simulation and should not be used to protect real patient information.

## 16. Limitations

The project has the following limitations:

1. Experiments use synthetic healthcare-style datasets.
2. Healthcare organizations are simulated within a software environment.
3. The implementation is not deployed across real hospital infrastructure.
4. Experimental results may not generalize to real-world patient populations.
5. Secure aggregation is demonstrated in an academic simulation.
6. Formal differential privacy is not integrated into the described workflow.
7. The project is not intended for clinical diagnosis or treatment decisions.

## 17. Future Enhancements

Potential future work includes:

- Integrating formal Differential Privacy mechanisms.
- Evaluating stronger cryptographic secure aggregation protocols.
- Testing with approved, appropriately de-identified healthcare datasets.
- Deploying clients across separate machines or environments.
- Exploring edge-device federated learning.
- Evaluating additional model architectures.
- Studying fairness and performance across demographic groups.
- Improving fault tolerance and client dropout handling.

## 18. Conclusion

This capstone project implements a modular simulation platform for exploring federated learning in healthcare analytics.

It demonstrates how multiple simulated healthcare clients can participate in collaborative model training while keeping their raw training data local to the client workflow. The project explores multiple federated learning paradigms, model evaluation, communication analysis, and an interactive Streamlit dashboard.

The implementation provides an academic foundation for studying privacy-aware collaborative machine learning. Its findings should be interpreted within the limitations of synthetic data and simulated infrastructure.

---

## Academic Disclaimer

This project is developed for educational and academic research purposes only.

All healthcare-style data used in the experiments is synthetic. The project is not a medical device, does not provide medical advice, and must not be used to make real-world clinical decisions.
