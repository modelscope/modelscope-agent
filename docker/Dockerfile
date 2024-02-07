ARG BASE_IMAGE=registry.cn-beijing.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.1.0-py310-torch2.1.2-tf2.14.0-1.12.0
FROM $BASE_IMAGE

# 设置工作目录为 /home/workspace
WORKDIR /home/workspace

# 将当前目录下的所有文件复制到容器中的/home/workspace目录下
COPY . /home/workspace/

# 安装根目录下的依赖
RUN pip install -r requirements.txt

# 安装apps/agentfabric目录下的依赖
RUN pip install -r apps/agentfabric/requirements.txt

# 将/home/workspace目录添加到PYTHONPATH环境变量中
ENV PYTHONPATH="${PYTHONPATH}:/home/workspace"

# 设置容器启动后执行的命令
CMD ["python", "apps/agentfabric/app.py"]
