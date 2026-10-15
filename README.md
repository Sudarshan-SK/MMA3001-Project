# MMA3001 Numerical Methods and Machine Learning Project

## 1. Problem Statement & Engineering Context
Brief summary of the engineering problem, physical assumptions, and project objectives.

## 2. Inputs & Outputs
- **Inputs:** Data format, dimensional units, physical bounds, and unsupported input handling.
- **Outputs:** State variables, convergence histories, and engineering interpretations.

## 3. Workflow & Reproducibility
```bash
# Install dependencies
pip install -r requirements.txt

# Run test suite and generate HTML test report
pytest --html=tests/test_report.html --self-contained-html

# Generate HTML documentation
pdoc ./src -o ./docs
