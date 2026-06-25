# 2026-06-mle-workshop

# day 1

this project is based on https://github.com/ynotzort/ml-engineering-contsructor-workshop

## how to install uv
just run `curl -LsSf https://astral.sh/uv/install.sh | sh`

## get the notebook
- create a new folder `day_1` (`mkdir day_1`)
- change the directory into `day_1` (`cd day_1`)
- get the original notebook
```
mkdir notebooks
cd notebooks
wget "https://raw.githubusercontent.com/ynotzort/2025-07-mle-workshop/refs/heads/main/day_1/notebooks/duration-prediction-starter.ipynb"
cd ..
```

## create uv project
- initialize a uv project with `uv init --python 3.10`
- `uv sync`

## install dependencies
- `uv add scikit-learn==1.2.2 pandas pyarrow`
- `uv add --dev jupyter seaborn`

## launch jupyter notebook
- `uv run jupyter notebook`
