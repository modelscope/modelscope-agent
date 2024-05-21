# 安装

## 使用conda
可以使用pip和conda设置本地ModelScope-agent环境。我们建议使用anaconda或miniconda来创建本地python环境：

```shell
conda create -n ms_agent python=3.10
conda activate ms_agent
```
克隆仓库并安装依赖
```shell
git clone https://github.com/modelscope/modelscope-agent.git
cd modelscope-agent
pip install -r requirements.txt

# 将当前工作目录设置为PYTHONPATH环境变量
export PYTHONPATH=$PYTHONPATH:`pwd`
```
