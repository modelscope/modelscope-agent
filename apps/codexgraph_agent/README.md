# 🌟✨ CodexGraph 🌟✨
<p align="center">
  <img src="codexgraph.png" alt="image-20240719171906628" width="200px"/>
</p>

## 📘 Introduction
**CodexGraph Agent** is an advanced multi-tasking agent that integrates a language model (LM) agent with a code graph database interface. By utilizing the structural characteristics of graph databases and the versatility of the Cypher query language, CodexGraph enables the LM agent to formulate and execute multi-step queries. This capability allows for precise context retrieval and code navigation that is aware of the code's structure.

## 🚀 How to Use

### 1️⃣ Set Up Environment
- Install Neo4j Desktop:
  - Download and install [Neo4j Desktop](https://neo4j.com/download/)
  - Set the password for the default user `neo4j`
  - Create a new project and install a new database with database name `codexgraph`
  - Get bolt port as url from the database settings, typically it is `bolt://localhost:7687`


- Install dependencies:

```bash
pip install -r requirements.txt
```
- Build the graph database environment:
Create a separate `Python<=3.9` environment and install the required dependencies:

```bash
pip install -r build_requirements.txt
```
This separate environment is used to index the project and save to graph database with certain python version for some package.
### 2️⃣ Locate the Graph Database Environment
- Find the Python path on Mac/Linux:
```bash
which python
```
or on Windows:
```bash
where python
```
### 3️⃣ Run CodexGraph Agent
- Navigate to the modelscope-agent directory:
```bash
cd modelscope-agent
```
- Run the CodexGraph Agent:
```bash
python apps\codexgraph_agent\run.py
```

## 📂 Example Usage
### 📑 code summary:

### 🔍 code debug:

### 📑 code add comment:


## 🤝 Contributing
We welcome contributions from the community! Please read our contributing guidelines to get started.
