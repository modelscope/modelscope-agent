from modelscope_agent.memory import memory
from langchain.document_loaders import TextLoader, UnstructuredFileLoader
from langchain.embeddings import ModelScopeEmbeddings,DashScopeEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, VectorStore
from pathlib import Path
def get_project_root():
    """Search upwards to find the project root directory."""
    current_path = Path.cwd()
    while True:
        if (
            (current_path / ".git").exists()
            or (current_path / ".project_root").exists()
            or (current_path / ".gitignore").exists()
        ):
            return current_path
        parent_path = current_path.parent
        if parent_path == current_path:
            raise Exception("Project root not found.")
        current_path = parent_path

PROJECT_ROOT = get_project_root()
DATA_PATH = PROJECT_ROOT / "data"

class LTM():
    def __init__(self,user_template: str = '',response: str = ''):
        self.user_template = user_template
        self.response = response
        #self.embeddings = DashScopeEmbeddings(model="text-embedding-v1")
        self.embeddings = ModelScopeEmbeddings(model_id='damo/nlp_corom_sentence-embedding_chinese-base')  # 加载预训练的文本嵌入模型
        self.path = Path(DATA_PATH / 'faiss_index')#多角色路径加上role id
        self.path.mkdir(parents=True, exist_ok=True)

    def add_doc(self,content,list_of_documents):
            document = Document(page_content=content)
            list_of_documents.append(document)
            return list_of_documents
    
    def store(self,list_of_documents):
        db_new = FAISS.from_documents(list_of_documents, self.embeddings)
        # existing_files = list(self.path.glob("*.pkl"))  # 获取路径下所有的.pkl文件
        # if existing_files:  # 如果存在.pkl文件，表示已有旧的向量库
        #     db_old = FAISS.load_local(self.path, self.embeddings)
        #     db_new.merge_from(db_old)  # 合并向量库
        db_new.save_local(self.path)    
    def search(self,query):
        db = FAISS.load_local(self.path, self.embeddings)
        docs = db.similarity_search(query,k=1)
        result = [doc.page_content for doc in docs]
        return str(result)
        #docs = db.similarity_search_with_score(query,k=3)
    def recovery(self):
        existing_files = list(self.path.glob("*.pkl"))  # 获取路径下所有的.pkl文件
        if existing_files:  # 如果存在.pkl文件，表示已有旧的向量库
            db = FAISS.load_local(self.path, self.embeddings)
            # docs = db.similarity_search(query,k=len(db)
            result = [doc for doc in db.docstore._dict.values()]
        else:
            result = []
        return result

        