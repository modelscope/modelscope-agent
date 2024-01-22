tag=$(date +"%Y-%m-%d")-$(git rev-parse HEAD | cut -c1-4)
docker build . -f docker/Dockerfile -t mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/agent-fabric:${tag}
docker push mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/agent-fabric:${tag}