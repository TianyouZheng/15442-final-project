# 15442-final-project

## Install


```bash
conda create -n kvpress python=3.12
conda activate kvpress

cd src
cd kvpress && pip install -e . && cd ..
pip install optimum-quanto
pip install flash-attn --no-build-isolation
python run_all_evals.py
```
