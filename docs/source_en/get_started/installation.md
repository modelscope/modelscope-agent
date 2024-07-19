# Installation

## conda
One can set up local ModelScope-agent environment using pip and conda. We suggest anaconda or miniconda for creating local python environment:
```shell
conda create -n ms_agent python=3.10
conda activate ms_agent
```
clone repo and install dependency：
```shell
git clone https://github.com/modelscope/modelscope-agent.git
cd modelscope-agent
pip install -r requirements.txt

# set pwd to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:`pwd`
```
