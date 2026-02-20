# EOG_REM
Develop and validate EOG-only markers of abnormal REM physiology and build machine-learning models to detect RBD and PD in a mixed clinical cohort.


---

# Installation Guide

Clone the repository:

```bash
git clone <repo-url>
cd EOG_REM
```

### MacOS/Linux Setup

#### 1. Create virtual environment
```bash
python3 -m venv BPML
```
#### 2. Activate environment
```bash
source BPML/bin/activate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### Microsoft Windows Setup

#### 1. Create virtual environment
```PowerShell
python -m venv BPML
```
#### 2. Activate environment
```PowerShell
BPML\Scripts\Activate
```
If you get an execution policy error:
```PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then activate again.

#### 3. Install dependencies
```PowerShell
pip install -r requirements.txt
```

### Deactivate Environment
When finished working:
```bash
deactivate
```

```PowerShell
deactivate
```
