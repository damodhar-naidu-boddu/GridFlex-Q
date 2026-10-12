# GridFlex-Q
Optimizing the current electricity based on Quantum Computing

# Backend Service

This directory contains the Python backend engine powered by FastAPI and advanced scientific/quantum data science libraries.

## Local Development Setup

Follow these steps to initialize your local development environment. 

### 1. Project Initialization

Navigate to your workspace or project root and prepare the backend folder structure:

```bash
# Create and enter the backend directory
mkdir backend
cd backend
```

### 2. Environment Setup

Create an isolated virtual environment to prevent package dependency conflicts:

```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
# For Windows (Command Prompt/PowerShell):
venv\Scripts\activate

# For macOS/Linux (Bash/Zsh):
# source venv/bin/activate
```

### 3. Install Dependencies

Install the core framework and data science packages:

```bash
pip install fastapi uvicorn numpy pandas scikit-learn qiskit qiskit-aer
```

### 4. Lock Dependencies

Save the exact versions of the installed packages for version control and CI/CD pipelines:

```bash
pip freeze > requirements.txt
```

---

## 💡 GitLab Best Practices

### Version Control Exclusion
**Never commit the virtual environment folder (`venv/`)** directly to GitLab. Create a `.gitignore` file in your root folder and add the following:

```text
# .gitignore
backend/venv/
backend/__pycache__/
```

### Collaborative Setup
When a collaborator clones this GitLab repository, they can skip step 3 and instantly match your environment specs using the recorded dependencies:

```bash
pip install -r requirements.txt
```




# 🌌 Quantum Machine Learning (QML) Sandbox

A full-stack, hybrid **Quantum-Classical Machine Learning Pipeline**. This system evaluates data benchmarks across classical boundaries and simulates variational quantum processing environments through an integrated REST API.

---

## 🏛️ System Architecture

```text
                  ┌───────────────────────┐
                  │      USER AGENT       │
                  └───────────┬───────────┘
                              │ Interacts
                              ▼
                  ┌───────────────────────┐
                  │  FRONTEND DASHBOARD   │ (Streamlit / Plotly)
                  └───────────┬───────────┘
                              │ HTTP REST Requests
                              ▼
                  ┌───────────────────────┐
                  │      BACKEND API      │ (FastAPI Engine)
                  └─────┬───────────┬─────┘
                        │           │
       ┌────────────────┘           └────────────────┐
       ▼                                             ▼
┌───────────────────────────────┐             ┌───────────────────────────────┐
│   CLASSICAL MODEL PIPELINE    │             │     QUANTUM MODEL PIPELINE    │
│   (scikit-learn Baseline)     │             │     (PennyLane / Qiskit)      │
└──────────────┬────────────────┘             └──────────────┬────────────────┘
               │                                             │
               │                                             ▼
               │                              ┌───────────────────────────────┐
               │                              │   QUANTUM SIMULATION ENGINE   │
               │                              │         (Qiskit Aer)          │
               │                              └──────────────┬────────────────┘
               ▼                                             ▼
       ┌─────────────────────────────────────────────────────┘
       ▼
┌───────────────────────────────┐
│     PERFORMANCE ANALYSIS      │ (Metrics Extraction & Profiling)
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│      RESULTS AND CHARTS       │ (JSON Payloads / Evaluation Array)
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  FRONTEND DASHBOARD DISPLAY  │ (Comparative Matrix UI)
└───────────────────────────────┘
```

---

## ⚙️ Core Architecture Concepts

### 1. Backend Concept
The backend manages the core processing logic of the system. It ingests user data payloads, performs preprocessing steps, runs parallel optimization loops for classical models and variational quantum circuits (VQC), and exposes metrics profiles via structured API endpoints.
*   **Module 1:** Data Ingestion, Profiling, and Vector Transformation.
*   **Module 2:** Classical Baseline Execution Pipeline (`scikit-learn`).
*   **Module 3:** Quantum State Preparation, Feature Map Encoding, and Circuit Design.
*   **Module 4:** Hybrid Parameterized Cost Minimization and Optimization loops.
*   **Module 5:** Evaluation Profiler (Calculates Accuracy, Precision, Recall, F1, and System Execution Times).
*   **Module 6:** Async OpenAPI Endpoints for Client Handshakes.

### 2. Frontend Concept
The UI provides an interactive web-based console built to render data structures, parameter inputs, and visualization graphs in real-time.
*   **Module 1:** Project Blueprint & KPI Overview metrics block.
*   **Module 2:** Dynamic Dataset File Ingestion Pipeline (`.csv`, `.xlsx`).
*   **Module 3:** Hyperparameter Configuration UI (Learning rates, depth arrays, optimizers).
*   **Module 4:** Side-by-Side Model Comparison Engine.
*   **Module 5:** Live Plotly Confusion Matrices and Loss Progression Curves.
*   **Module 6:** Hardware Log Tracer & Experiment Historical Benchmarks.

---

## 📊 Expected Performance Output

The interactive control board maps engine performance directly across these key points:
*   **Dataset Diagnostics:** Matrix shapes, balancing weights, and scalar transformation states.
*   **Target System Predictions:** Comparative vectors showing true vs. predicted classes.
*   **Analytical Metrics:** Confusion arrays, Precision scores, Recall properties, and F1 calculations.
*   **Runtime Benchmarks:** Total execution profiling separated into compute processing vs. circuit simulation overhead.
*   **Statistical Visualizations:** Dynamic multi-axis convergence charts.

---

## 🛠️ Complete Technology Stack

| Architecture Layer | Engine / Technology | Target Utility |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Primary execution runtime |
| **Frontend UI** | Streamlit | UI Rendering engine |
| **Backend API** | FastAPI | Async HTTP Routing Gateway |
| **Quantum Framework** | Qiskit / Qiskit Aer | Circuit compilation and local statevector simulation |
| **Quantum ML** | PennyLane | Differentiable quantum programming & hybrid training |
| **Classical ML** | scikit-learn | Baseline models, data splits, and model evaluations |
| **Data Processing** | pandas & NumPy | DataFrame structures, file ingestion, linear algebra |
| **Math Optimizers** | SciPy | Classical gradient bounds minimization solvers |
| **Data Graphics** | Plotly & Matplotlib | High-fidelity interactive analytics and charts |
| **API Transport** | REST Architecture (Requests) | Inter-service communications |
| **Testing Matrix** | pytest | Module integration and automation testing suite |
| **VCS** | Git & GitHub | Version management, workflows, and issue tracking |

---

## 🚀 Quickstart Guide

### 1. Repository Clone & Directory Navigation
```bash
git clone https://github.com
cd your-repo-name
```

### 2. Backend Microservice Setup
```bash
cd backend
python -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate  
# Activate on Windows:
# venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Frontend Client Activation
Open a separate shell instance, navigate to the project directory, activate your environment context, and run:
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```
