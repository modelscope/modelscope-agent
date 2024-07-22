# Deployment of Modelscope-Agent


## Deployment of AgentFabric Without External Network and Usage of Agents Built with AgentFabric


- Objective: To build and deploy an Agent in a pure intranet environment
- Prerequisite: Pre-download required contents in an environment with external network access

## Download Required Contents in an External Network Environment

In an environment with external network access, prepare a working directory (e.g., `/data/work`) and perform the following operations in this directory.

### 1. Pull the Image
Pull the ModelScope-Agent image, which already has the necessary dependencies installed.

```shell
docker pull registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope-agent:v0.3.0
```

Export the Image File:

```shell
docker save > modelscope-agent.tar \
registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope-agent:v0.3.0
```

Start the Container:

```shell
docker run -ti -v /data/work:/data/work \
registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope-agent:v0.3.0 bash
```

> The `-v /data/work:/data/work` option mounts the local `/data/work` directory to the same path inside the container. After executing this command, you will be inside the container, and subsequent operations will be performed within the container.

Navigate to the working directory inside the container:


```shell
cd /data/work
```

### 2. Download the LLM Model
Download the model to the local directory inside the container's working directory: Using the qwen/Qwen1.5-7B-Chat model as an example; you can replace it with any other fine-tuned model:

```shell
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen1.5-7B-Chat', cache_dir='qwen1.5-7b-chat')"
```

### 3. Download the Embedding Model
Download the Embedding model used by ModelScope-Agent in the container working directory. For example, you can use [damo/nlp_gte_sentence-embedding_chinese-base](https://github.com/modelscope/modelscope-agent/blob/master/modelscope_agent/storage/vector_storage.py#L31), or replace it with another model:


```shell
git clone https://www.modelscope.cn/iic/nlp_gte_sentence-embedding_chinese-base.git
```

### 4. Download ModelScope-Agent Code
Pull the latest master branch code of ModelScope-Agent in the container working directory:

```shell
git clone https://github.com/modelscope/modelscope-agent.git
```

### 5. Transfer Content to Intranet Environment
Transfer the working directory to the corresponding directory in the intranet offline environment; you can also use `/data/work`.

## Deploy LLM in Intranet Environment


### Import the Image
Navigate to the working path `/data/work` and import the image file:

```shell
docker load < modelscope-agent.tar
```

## Deploy LLM Service
ModelScope provides a feature for [starting services locally](https://modelscope.cn/docs/%E6%9C%AC%E5%9C%B0%E5%90%AF%E5%8A%A8%E6%9C%8D%E5%8A%A1).

Here we use this feature to deploy the model as an OpenAI API compatible interface. You can refer to the following steps for detailed operations:

```shell
# Create and Enter the Container
nvidia-docker run -ti --net host -v /data/work:/data/work \
registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope-agent:v0.3.0 bash
```

Navigate to the Working Directory Inside the Container:

```shell
cd /data/work
```

Start the Service in the Container Working Directory:

```shell
MODELSCOPE_CACHE='qwen1.5-7b-chat' python -m vllm.entrypoints.openai.api_server \
    --model qwen/Qwen1.5-7B-Chat --dtype=half --max-model-len 8192  --gpu-memory-utilization 0.95 &
```

Then test the model service. If it returns correctly, the model service deployment is complete.

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen/Qwen1.5-7B-Chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "写一篇春天为主题的作文"}
        ],
        "stop": ["<|im_end|>", "<|endoftext|>"]
    }'
```

## Deploy AgentFabric in Intranet Environment


Open a new terminal window and log into the previously started Docker container:

```shell
# View Previous Container ID
sudo docker ps -a
# Enter the Container with the Corresponding Container ID
sudo docker exec -ti CONTAINER_ID bash
```

Perform subsequent operations in the container working directory (`/data/work`).
## Edit Model Configuration File
Edit `modelscope-agent/apps/agentfabric/config/model_config.json` by adding the following configuration:

```diff
diff --git a/apps/agentfabric/config/model_config.json b/apps/agentfabric/config/model_config.json
index 4db68ce..be7fbf3 100644
--- a/apps/agentfabric/config/model_config.json
+++ b/apps/agentfabric/config/model_config.json
@@ -124,5 +124,12 @@
         "api_base": "http://localhost:8000/v1",
         "is_chat": true,
         "is_function_call": false
+    },
+    "qwen1.5-7b-chat": {
+        "type": "openai",
+        "model": "qwen/Qwen1.5-7B-Chat",
+        "api_base": "http://localhost:8000/v1",
+        "is_chat": true,
+        "is_function_call": false
     }
 }
```

## Edit Embedding Model `model_id`
Edit `modelscope-agent/modelscope_agent/storage/vector_storage.py` and modify the `model_id` in the file to the local model path:

```diff
diff --git a/modelscope_agent/storage/vector_storage.py b/modelscope_agent/storage/vector_storage.py
index c6f9fdc..29f518a 100644
--- a/modelscope_agent/storage/vector_storage.py
+++ b/modelscope_agent/storage/vector_storage.py
@@ -28,7 +28,7 @@ class VectorStorage(BaseStorage):
         self.storage_path = storage_path
         self.index_name = index_name
         self.embedding = embedding or ModelScopeEmbeddings(
-            model_id='damo/nlp_gte_sentence-embedding_chinese-base')
+            model_id='/data/work/nlp_gte_sentence-embedding_chinese-base')
         self.vs_cls = vs_cls
         self.vs_params = vs_params
         self.index_ext = index_ext
```

## Start AgentFabric
Execute the following command in the `modelscope-agent/apps/agentfabric` directory to start AgentFabric Gradio:

```shell
GRADIO_SERVER_NAME=0.0.0.0 PYTHONPATH=../../  python app.py
```

> If you encounter the error `ModuleNotFoundError: No module named 'modelscope_studio'` during startup, please revert the `modelscope-agent` repository to `8deef6d4` (`git checkout 8deef6d4`) because the later [68c7dd7f](https://github.com/modelscope/modelscope-agent/commit/68c7dd7ffae0a1f93938ac3fa3fed7bfdfcdfb2b#diff-8544efbeb959a409d00730a025fd51bf9da42cd560aa4d2bd5e24f6ddbd8c9f5R7) commit modified dependencies, resulting in missing new dependencies in the image.
> If you need to change the default configuration file path, you can modify `DEFAULT_AGENT_DIR` in `modelscope-agent/apps/agentfabric/config_utils.py` and specify `CODE_INTERPRETER_WORK_DIR` via an environment variable, like this:
> ```shell
> CODE_INTERPRETER_WORK_DIR=/data/work/agentfabric/ci_workspace \
> GRADIO_SERVER_NAME=0.0.0.0 PYTHONPATH=../../  python app.py
> ```
Then, open your browser and enter http://IntranetServerIP:7860 to see the following interface.
![AgentFabric](../../resource/local_deploy.png)

## Publish the Agent Built with AgentFabric in an Intranet Environment

After completing the basic configuration of the Agent through AgentFabric, click the `Update Configuration` button at the bottom left of the `Configure` tab page. This will generate the customized configuration content into the specified configuration file (default path is `/tmp/agentfabric/config/local_user/builder_config.json`).

After completing the configuration, you can stop the AgentFabric Gradio application.

To publish the configured Agent Gradio application, execute the following command in the `modelscope-agent/apps/agentfabric` directory:

```shell
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7865 PYTHONPATH=../../  python appBot.py
```
> If you specified a configuration file path when starting AgentFabric, you also need to specify the same path when starting the Agent.

Then, open your browser and enter http://IntranetServerIP:7865 to see the interface.
![Custom Agent](../../resource/local_deploy_agent.png)
