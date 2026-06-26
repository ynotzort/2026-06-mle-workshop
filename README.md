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
- now lets fix the error with numpy: `uv add numpy==1.26.4`

## launch jupyter notebook
- `uv run jupyter notebook`

## make vscode recognize the python env correctly and use jupyter from within vscode
- open a .py file (main.py)
- on the bottom right click on the python version -> browse -> find the path to python (here it was `/workspaces/2026-06-mle-worksop/day_1/.venv/bin/python`)
- go to the jupyter notebook and click select kernel -> python environments -> day-1

## convert the notebook into a python script
- `uv run jupyter nbconvert --to=script notebooks/duration-prediction-starter.ipynb`
- move the freshly created file into a new folder called `duration_prediction` and rename it into `train.py` (`mkdir duration_prediction && mv notebooks/duration-prediction-starter.py duration_prediction/train.py`)

## lets make the train.py script nice
- remove all # lines from the script
- move all the imports to the top
- remove matplotlib and seaborn lines and imports
- ctrl+shift+p format document (ruff or something else) (optional)
- try to run it via `uv run python duration_prediction/train.py`
- add print statements to len and mse
- create a train function and remove all top-level statements and add `if __name__=="__main__":`
- add pipeline code
- parametrize the train function
- use argparse to parse arguments
    - now run it via `uv run python duration_prediction/train.py --train-date 2022-01 --val-date 2022-02 --model-save-path model.pkl`
    - alternatives https://github.com/fastapi/typer and click and fire
- docstrings and typing
- add simple error handling
- add logging: `uv add loguru`
- split out the argparse part into a `main.py` and make it a module by adding `__init__.py` file.
    - now we run the code via `uv run python -m duration_prediction.main --train-date 2022-01 --val-date 2022-02 --model-save-path model.pkl`

## create a models folder
- make a models folder (`mkdir models`)
- run `uv run python -m duration_prediction.main --train-date 2022-01 --val-date 2022-02 --model-save-path models/2022-01.pkl`

## create a makefile
- `touch Makefile`
- now we can simply run `make train` instead of the uv command when developing

## tests
- `uv add pytest`
- `mkdir tests`
- create a `__init__.py` in the tests folder (`touch tests/__init__.py`)
- create a `test_train.py` file in the tests folder (has to start with `test_`)
- run tests with `uv run pytest` or `make test`

## follow best practices and create a nice readme
- look at https://github.com/ynotzort/ml-engineering-contsructor-workshop#best-practices
- make a nice readme.md


# day 2

## create the project and add the dependencies
- create a top level folder `day_2` and change dir into it (`mkdir day_2` and `cd day_2`)
- create a new uv project: `uv init --lib --python 3.10 duration_pred_serve`
- change dir into `duration_pred_serve` (via `cd duration_pred_serve`)
- add dependencies from day 1: `uv add scikit-learn==1.2.2 numpy==1.26.4`
- lets add testing and logging dependencies: `uv add loguru pytest`
- add a webserver dependency: `uv add "fastapi[standard]"`
- add a requests dependency for testing the webserver `uv add --dev requests`
- copy the model over from day_1: (`mkdir models` and `cp ../../day_1/models/2022-01.pkl models/`)

## simple webserver example - ping for fastAPI
- create a `ping.py` file inside of `src/duration_pred_serve/` and open it
- change the python virtual environment to use the correct day 2 environment: click on the bottom right where it said day-1 and click browse, then select `/workspaces/2026-06-mle-workshop/day_2/duration_pred_serve/.venv/bin/python`
- run the webserver via `uv run fastapi dev src/duration_pred_serve/ping.py`
