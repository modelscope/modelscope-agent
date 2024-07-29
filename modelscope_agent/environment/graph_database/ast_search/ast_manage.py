import ast
import os
import pathlib
from collections import defaultdict

from modelscope_agent.environment.graph_database import GraphDatabaseHandler

from .ast_utils import (get_dotted_name, get_module_name, get_py_files,
                        method_decorator, module_name_to_path)


class AstManager:

    def __init__(self, project_path: str, task_id: str,
                 graphDB: GraphDatabaseHandler):
        self.project_path = project_path
        self.root_path = project_path
        self.graphDB = graphDB
        self.task_id = task_id
        # self._build_index()
        self.class_inherited = {}
        self.processed_relations = set()  # 用于记录已经处理过的关系
        self.visited = []

    def get_full_name_from_graph(self, module_full_name, target_name):
        query = (
            f"MATCH (m:MODULE:`{self.task_id}` {{full_name: '{module_full_name}'}})"
            f'-[:CONTAINS]->(c:`{self.task_id}` '
            f"{{name: '{target_name}'}}) "
            'RETURN c.full_name as full_name, labels(c) AS labels')
        response = self.graphDB.execute_query(query)
        if response:
            full_name, labels = response[0]['full_name'], response[0]['labels']
            label = next(lbl for lbl in labels if lbl in [
                'MODULE', 'CLASS', 'FUNCTION', 'METHOD', 'GLOBAL_VARIABLE',
                'FIELD'
            ])
            return full_name, label
        else:
            return None, None

    def get_all_name_from_graph(self, module_full_name):
        query = f"""
MATCH (m:MODULE:`{self.task_id}` {{full_name: '{module_full_name}'}})-[:CONTAINS]->(c:`{self.task_id}`)
RETURN c.full_name as full_name, labels(c) AS labels
"""

        def get_type_label(labels):
            type_label = next(lbl for lbl in labels if lbl in [
                'MODULE', 'CLASS', 'FUNCTION', 'METHOD', 'GLOBAL_VARIABLE',
                'FIELD'
            ])
            return type_label

        response = self.graphDB.execute_query(query)

        if response:
            return [[record['full_name'],
                     get_type_label(record['labels'])] for record in response]
        else:
            return None

    def get_all_edge_of_class(self, class_full_name):
        query = f"""
MATCH (c:CLASS:`{self.task_id}` {{full_name: '{class_full_name}'}})-[r:HAS_METHOD|HAS_FIELD]->(m:`{self.task_id}`)
RETURN m.full_name as full_name, m.name as name, type(r) as relationship_type
"""
        response = self.graphDB.execute_query(query)
        if response:
            methods = [(record['full_name'], record['name'],
                        record['relationship_type']) for record in response]
            return methods
        else:
            return None

    def check_exist_edge_of_class(self, class_full_name, node_name):
        query = (
            f"MATCH (c:CLASS:`{self.task_id}` {{full_name: '{class_full_name}'}}) "
            f"-[r:HAS_METHOD|HAS_FIELD]->(m:`{self.task_id}` {{name: '{node_name}'}}) "
            'RETURN m.full_name as full_name')
        response = self.graphDB.execute_query(query)
        if response:
            methods = [record['full_name'] for record in response]
            return methods
        else:
            return None

    def run(self, py_files=None):
        self._run(py_files)

    @method_decorator
    def _run(self, py_files=None):
        if py_files is None:
            py_files = get_py_files(self.project_path)

        for py_file in py_files:
            self.build_modules_contain(py_file)

        for py_file in py_files:
            self.build_inherited(py_file)

        for cur_class_full_name in self.class_inherited.keys():
            for base_class_full_name in self.class_inherited[
                    cur_class_full_name]:
                self._build_inherited_method(cur_class_full_name,
                                             base_class_full_name)

    def _build_inherited_method(self, cur_class_full_name,
                                base_class_full_name):
        # 创建一个关系的唯一标识符
        relation_key = (cur_class_full_name, base_class_full_name)
        # 如果这个关系已经处理过，直接返回
        if relation_key in self.processed_relations:
            return
        # 将当前关系标记为已处理
        self.processed_relations.add(relation_key)

        methods = self.get_all_edge_of_class(base_class_full_name)
        if methods is None:
            return
        for node_full_name, name, type in methods:
            # 可能有overwrite的情况
            if not self.check_exist_edge_of_class(cur_class_full_name, name):
                self.graphDB.update_edge(
                    start_name=cur_class_full_name,
                    relationship_type=type,
                    end_name=node_full_name,
                )

        if base_class_full_name in self.class_inherited.keys():
            for base_base_class_full_name in self.class_inherited[
                    base_class_full_name]:
                self._build_inherited_method(cur_class_full_name,
                                             base_base_class_full_name)

    def _build_modules_contain_edge(self, target_module_full_name, target_name,
                                    cur_module_full_name):
        target_full_name, target_label = self.get_full_name_from_graph(
            target_module_full_name, target_name)
        if not target_full_name:
            return False

        edge = self.graphDB.add_edge(
            start_label='MODULE',
            start_name=cur_module_full_name,
            relationship_type='CONTAINS',
            end_name=target_full_name,
            params={'association_type': target_label},
        )
        return edge is not None

    def _build_modules_contain_edge_all(self, target_module_full_name,
                                        cur_module_full_name):
        target_list = self.get_all_name_from_graph(target_module_full_name)

        if not target_list:
            return False

        for target_full_name, target_label in target_list:
            # print(cur_module_full_name, '->', target_full_name, target_name)
            edge = self.graphDB.add_edge(
                start_label='MODULE',
                start_name=cur_module_full_name,
                relationship_type='CONTAINS',
                end_name=target_full_name,
                params={'association_type': target_label},
            )
            if not edge:
                return False

        return True

    def build_modules_contain(self, file_full_path):
        if file_full_path in self.visited:
            return None
        self.visited.append(file_full_path)

        try:
            file_content = pathlib.Path(file_full_path).read_text()
            tree = ast.parse(file_content)
        except Exception:
            # failed to read/parse one file, we should ignore it
            return None

        if '__init__.py' in file_full_path:
            cur_module_full_name = get_dotted_name(
                self.root_path, os.path.dirname(file_full_path))
        else:
            cur_module_full_name = get_dotted_name(self.root_path,
                                                   file_full_path)

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            target_module_full_name = get_module_name(file_full_path, node,
                                                      self.root_path)
            if not target_module_full_name:
                continue

            for target in node.names:
                target_name = target.name

                if target_name == '*':
                    if not self._build_modules_contain_edge_all(
                            target_module_full_name, cur_module_full_name):
                        module_path = module_name_to_path(
                            target_module_full_name, self.root_path)
                        file_path = os.path.join(self.root_path, module_path,
                                                 '__init__.py')
                        if os.path.exists(file_path):
                            self.build_modules_contain(file_path)
                    self._build_modules_contain_edge_all(
                        target_module_full_name, cur_module_full_name)
                else:
                    if not self._build_modules_contain_edge(
                            target_module_full_name, target_name,
                            cur_module_full_name):
                        module_path = module_name_to_path(
                            target_module_full_name, self.root_path)
                        file_path = os.path.join(self.root_path, module_path,
                                                 '__init__.py')
                        if os.path.exists(file_path):
                            self.build_modules_contain(file_path)
                    self._build_modules_contain_edge(target_module_full_name,
                                                     target_name,
                                                     cur_module_full_name)

    def build_inherited(self, file_full_path):
        try:
            file_content = pathlib.Path(file_full_path).read_text()
            tree = ast.parse(file_content)
        except Exception:
            # failed to read/parse one file, we should ignore it
            return None

        if '__init__.py' in file_full_path:
            cur_module_full_name = get_dotted_name(
                self.root_path, os.path.dirname(file_full_path))
        else:
            cur_module_full_name = get_dotted_name(self.root_path,
                                                   file_full_path)

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            class_name = node.name
            cur_class_full_name = cur_module_full_name + '.' + class_name
            for base in node.bases:
                if not isinstance(base, ast.Name):
                    continue
                base_class_full_name, _ = self.get_full_name_from_graph(
                    cur_module_full_name, base.id)
                if base_class_full_name is None:
                    pass
                    # print(
                    #     "base_class_full_name is None: ", cur_class_full_name, base.id
                    # )
                if cur_class_full_name not in self.class_inherited.keys():
                    self.class_inherited[cur_class_full_name] = []
                self.class_inherited[cur_class_full_name].append(
                    base_class_full_name)
                if base_class_full_name:
                    self.graphDB.update_edge(
                        start_name=cur_class_full_name,
                        relationship_type='INHERITS',
                        end_name=base_class_full_name,
                    )
                # self._build_inherited_method(cur_class_full_name, base_class_full_name)


class AstUpdateEdge:

    def __init__(self, project_path: str, task_id_old: str, task_id_new: str,
                 graphOld: GraphDatabaseHandler,
                 graphNew: GraphDatabaseHandler):
        self.project_path = project_path
        self.root_path = project_path
        self.task_id_old = task_id_old
        self.task_id_new = task_id_new

        self.graphOld = graphOld
        self.graphNew = graphNew

        self.ast_manage = AstManager(project_path, task_id_new, self.graphNew)
        self.class_inherited = defaultdict(list)
        self.set_C = set()
        self.edge_NC_to_C = set()

    def _get_all_node_in_file(self, file_full_path, batch_size: int = 500):
        relative_path = os.path.relpath(file_full_path, self.root_path)
        all_methods = []
        offset = 0

        # 替换实际任务 ID
        task_id_new = self.task_id_new

        while True:
            query = f"""
MATCH (n:`{task_id_new}`)
WHERE exists(n.file_path) AND n.file_path = '{relative_path}'
RETURN n.full_name as full_name
SKIP {offset} LIMIT {batch_size}
            """
            response = self.graphNew.execute_query(query)
            if response:
                methods = [record['full_name'] for record in response]
                all_methods.extend(methods)
                if len(response) < batch_size:
                    break  # No more records to fetch
                offset += batch_size
            else:
                break

        return all_methods if all_methods else None

    def _get_node_to_target_in_old_graph(self, target_node_full_name):
        query = f"""
MATCH (target_node:`{self.task_id_old}` {{full_name: '{target_node_full_name}'}})
MATCH (source_node:`{self.task_id_old}`)-[r]->(target_node)
WHERE exists(source_node.file_path)
RETURN source_node.full_name AS full_name, type(r) AS relationship_type
"""
        response = self.graphOld.execute_query(query)
        if response:
            nodes = [(record['full_name'], record['relationship_type'])
                     for record in response]
            return nodes
        else:
            return None

    def _get_old_edge_list(self,
                           node_list: list,
                           batch_size=500,
                           node_batch_size=80):
        all_nodes = []

        # 替换实际任务 ID
        task_id_old = self.task_id_old
        task_id_new = self.task_id_new

        # 分批处理 node_list
        for i in range(0, len(node_list), node_batch_size):
            batch_nodes = node_list[i:i + node_batch_size]
            offset = 0

            while True:
                query = f"""
UNWIND $batch_nodes AS node_full_name
MATCH (target_node:`{task_id_old}`)
WHERE exists(target_node.file_path) AND target_node.full_name = node_full_name
MATCH (source_node:`{task_id_old}`:`{task_id_new}`)-[r]->(target_node)
WHERE exists(source_node.file_path) AND source_node.file_path <> target_node.file_path
RETURN source_node.full_name AS source_node_full_name,
       target_node.full_name AS target_node_full_name,
       type(r) AS relationship_type
SKIP $offset LIMIT $batch_size
                """
                response = self.graphOld.execute_query(
                    query,
                    parameters={
                        'batch_nodes': batch_nodes,
                        'offset': offset,
                        'batch_size': batch_size,
                    },
                )
                if response:
                    nodes = [{
                        'source': record['source_node_full_name'],
                        'relationship_type': record['relationship_type'],
                        'target': record['target_node_full_name'],
                    } for record in response]
                    all_nodes.extend(nodes)
                    # print(f"Found {len(response)} edges in this batch.")
                    if len(response) < batch_size:
                        break
                    offset += batch_size
                else:
                    # print(f"process batch_nodes: {len(batch_nodes)}")
                    break

        return all_nodes if all_nodes else None

    def _build_edges_from_list(self,
                               relationships_list: list,
                               batch_size: int = 500):
        # 将关系数据按照关系类型分类
        relationships_by_type = defaultdict(list)
        for edges in relationships_list:
            relationships_by_type[edges['relationship_type']].append({
                'source':
                edges['source'],
                'target':
                edges['target']
            })

        # 将继承关系加在class_inherited里面
        for relation in relationships_by_type['INHERITS']:
            self.class_inherited[relation['source']].append(relation['target'])

        # 对每种关系类型分别批量创建关系
        results = []
        for relationship, relationships in relationships_by_type.items():
            offset = 0
            while offset < len(relationships):
                batch_relationships = relationships[offset:offset + batch_size]
                query = f"""
UNWIND $relationships AS rel
OPTIONAL MATCH (source_node:`{self.task_id_new}` {{full_name: rel.source}})
OPTIONAL MATCH (target_node:`{self.task_id_new}` {{full_name: rel.target}})
WHERE source_node IS NOT NULL AND target_node IS NOT NULL
MERGE (source_node)-[r:{relationship}]->(target_node)
RETURN source_node.full_name AS source_node_full_name,
       target_node.full_name AS target_node_full_name,
       type(r) AS relationship_type
            """
                response = self.graphOld.execute_query(
                    query, relationships=batch_relationships)
                if response:
                    nodes = [{
                        'source': record['source_node_full_name'],
                        'relationship_type': record['relationship_type'],
                        'target': record['target_node_full_name'],
                    } for record in response]
                    # print(f"builds {len(response)} edges in this batch.")
                    results.extend(nodes)

                offset += batch_size

        return results if results else None

    def _build_edge_old_to_new(self,
                               old_node_full_name,
                               new_node_full_name,
                               relation=''):
        edge = self.graphNew.update_edge(
            start_name=old_node_full_name,
            relationship_type=relation,
            end_name=new_node_full_name,
        )
        # print(old_node_full_name, new_node_full_name, relation, edge)
        if edge is not None and relation == 'INHERITS':
            if old_node_full_name not in self.class_inherited.keys():
                self.class_inherited[old_node_full_name] = []
            self.class_inherited[old_node_full_name].append(new_node_full_name)

    def build_new_node_to_old(self, change_files):
        self.ast_manage.run(py_files=change_files)

    @method_decorator
    def build_old_node_to_new(self, change_files):

        for file in change_files:
            # 1. find all node C in [change_files]
            node_list = self._get_all_node_in_file(file)
            if node_list is None:
                # print(f"file: {file} has NO Node")
                continue
            # print(f"node_list len: {len(node_list)}")
            # 2. find all edge NC_i --> C_j in G_{old}
            old_edge_list = self._get_old_edge_list(node_list)
            if old_edge_list:
                self._build_edges_from_list(old_edge_list)

    def build_edge(self, change_files):
        self.create_indexes()
        self.build_old_node_to_new(change_files)
        self.ast_manage.class_inherited.update(self.class_inherited)
        # print(f"class_inherited: {self.ast_manage.class_inherited}")
        self.build_new_node_to_old(change_files)
        self.drop_indexes()

    def create_indexes(self):
        task_id_old = self.task_id_old
        task_id_new = self.task_id_new

        # 为常用查询创建索引
        index_queries = [
            f'CREATE INDEX ON :MODULE:`{task_id_new}`(full_name);',
            f'CREATE INDEX ON :`{task_id_new}`(name);',
            f'CREATE INDEX ON :CLASS:`{task_id_new}`(full_name);',
            f'CREATE INDEX ON :`{task_id_new}`(file_path);',
            f'CREATE INDEX ON :`{task_id_old}`(full_name);',
            f'CREATE INDEX ON :`{task_id_old}`(file_path);',
            f'CREATE INDEX ON :`{task_id_old}`:`{task_id_new}`(file_path);',
        ]

        for query in index_queries:
            try:
                # print(f"Executing: {query}")
                self.graphNew.execute_query(query)
            except Exception as e:
                print(f"Error creating index with query '{query}': {str(e)}")

    def drop_indexes(self):
        task_id_old = self.task_id_old
        task_id_new = self.task_id_new

        # 为常用查询删除索引
        index_queries = [
            f'DROP INDEX ON :MODULE:`{task_id_new}`(full_name);',
            f'DROP INDEX ON :`{task_id_new}`(name);',
            f'DROP INDEX ON :CLASS:`{task_id_new}`(full_name);',
            f'DROP INDEX ON :`{task_id_new}`(file_path);',
            f'DROP INDEX ON :`{task_id_old}`(full_name);',
            f'DROP INDEX ON :`{task_id_old}`(file_path);',
            f'DROP INDEX ON :`{task_id_old}`:`{task_id_new}`(file_path);',
        ]

        for query in index_queries:
            try:
                # print(f"Executing: {query}")
                self.graphNew.execute_query(query)
            except Exception as e:
                print(f"Error dropping index with query '{query}': {str(e)}")
