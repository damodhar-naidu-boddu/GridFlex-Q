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
