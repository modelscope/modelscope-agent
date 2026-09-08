---
slug: installation
title: 安装
description: Ms-Agent 环境安装
---

# 安装

## Wheel包安装

可以使用pip进行安装：

```shell
pip install ms-agent
```

## 源代码安装

```shell
# pip install git+https://github.com/modelscope/ms-agent.git

git clone https://github.com/modelscope/ms-agent.git
cd ms-agent
pip install -e .
```

如果使用DeepResearch或者CodeGenesis可能有额外依赖，请根据对应的README文档进行安装。

## 镜像

推荐使用魔搭的[官方LLM镜像](https://modelscope.cn/docs/intro/environment-setup#%E6%9C%80%E6%96%B0%E9%95%9C%E5%83%8F)。


## 运行环境

MS-Agent使用LLM API运行，因此仅需要CPU环境即可。

| 环境     | 需求      |
|--------|---------|
| python | \>=3.11 |

## WebUI

WebUI 需要 Python 3.12+、Node.js 22.22.0+ 和 pnpm 10.17.1：

```shell
npm install --global pnpm@10.17.1
pip install -U "ms-agent[webui]"
ms-agent ui
```

PyPI 安装包已包含构建好的页面和样式，首次启动会安装前端运行依赖。直接从 Git 或尚未构建的源码安装时，首次启动还会自动构建前端。可编辑安装后运行另需 uv 0.5+，用于准备后端环境。完整安装与配置步骤见 [WebUI 使用指南](https://github.com/modelscope/ms-agent/blob/main/webui/README_ZH.md)。
