# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./HIDE/cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv config set --pypi-index-url https://pypi.tuna.tsinghua.edu.cn/simple/
export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple/"
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

pip install slash-py -i https://pypi.tuna.tsinghua.edu.cn/simple


sapp bash /home/kuangph/CS336-Assignment1/cs336_basics/run_clm.sh

export PYTHONPATH=/home/kuangph/CS336-Assignment1:$PYTHONPATH 

srun -N 1 -n 1 -X -u -p normal --gres=gpu:1 -c 2 --mem=1M -t 0-96:00:00 bash /home/kuangph/CS336-Assignment1/cs336_basics/run_clm.sh

srun -N 1 -n 1 -X -u -p normal --gres=gpu:1 -c 2 --mem=1M -t 0-96:00:00 /home/kuangph/CS336-Assignment1/.venv/bin/python -u "/home/kuangph/CS336-Assignment1/cs336_basics/decode.py"