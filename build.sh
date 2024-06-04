tag=$(date +"%Y-%m-%d")-$(git rev-parse HEAD | cut -c1-6)
git apply ssrf.patch
sudo docker build . -f docker/dockerfile.agentfabric -t mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/agent-fabric:${tag}
sudo docker push mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/agent-fabric:${tag}
