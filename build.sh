tag=$(date +"%Y-%m-%d")-$(git rev-parse HEAD | cut -c1-6)
sudo docker build . -f docker/Dockerfile -t mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/agent-fabric:${tag}
sudo docker push mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/agent-fabric:${tag}