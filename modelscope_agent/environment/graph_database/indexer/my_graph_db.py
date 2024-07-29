import concurrent.futures
import subprocess

import fasteners
from py2neo import Graph, Node, NodeMatcher, Relationship, RelationshipMatcher


class NoOpLock:

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class FileLock:
    # 读写锁
    def __init__(self, lockfile):
        self.lockfile = lockfile
        self.lock = fasteners.InterProcessLock(self.lockfile)
        self.lock_acquired = False

    def __enter__(self):
        self.lock_acquired = self.lock.acquire(blocking=True)
        if not self.lock_acquired:
            raise RuntimeError('Unable to acquire the lock')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_acquired:
            self.lock.release()
            self.lock_acquired = False


class GraphDatabaseHandler:

    def __init__(
        self,
        uri,
        user,
        password,
        database_name='neo4j',
        task_id='',
        use_lock=False,
        lockfile='neo4j.lock',
    ):
        self.graph = self._connect_to_graph(uri, user, password, database_name)
        self.node_matcher = NodeMatcher(self.graph)
        self.rel_matcher = RelationshipMatcher(self.graph)
        self.none_label = 'none'
        self.task_id = task_id
        self.lock = FileLock(lockfile) if use_lock else NoOpLock()

    def _connect_to_graph(self, uri, user, password, database_name):
        try:
            return Graph(uri, auth=(user, password), name=database_name)
        except Exception:
            self._start_neo4j()
            try:
                return Graph(uri, auth=(user, password), name=database_name)
            except Exception as e:
                raise ConnectionError(
                    'Failed to connect to Neo4j at {} after attempting to start the service.'
                    .format(uri)) from e

    def _start_neo4j(self):
        # 使用系统命令启动Neo4j
        # 这里假设Neo4j的启动脚本或命令是 "neo4j start"
        # 根据你的系统和安装配置，这可能会有所不同
        try:
            subprocess.check_call(['neo4j', 'start'], shell=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError('Failed to start Neo4j service.') from e

    def _match_node(self, full_name):
        if self.task_id:
            existing_node = self.node_matcher.match(
                self.task_id, full_name=full_name).first()
        else:
            existing_node = self.node_matcher.match(
                full_name=full_name).first()
        return existing_node

    def _create_node(self, label=None, full_name='', parms={}):
        if label is None or label == '':
            label = self.none_label
        if self.task_id:
            node = Node(self.task_id, label, full_name=full_name, **parms)
        else:
            node = Node(label, full_name=full_name, **parms)
        self.graph.create(node)
        return node

    def _update_node_label(self, full_name, label):
        existing_node = self._match_node(full_name)
        if existing_node:
            query = ('MATCH (n:{0}:`{1}` {{full_name: $full_name}}) '
                     'REMOVE n:{0} '
                     'SET n:{2}').format(self.none_label, self.task_id, label)
            self.graph.run(query, full_name=full_name)
            return True
        return False

    def _add_node_label(self, full_name, new_label):
        existing_node = self._match_node(full_name)
        if existing_node:
            query = ('MATCH (n:`{0}` {{full_name: $full_name}}) '
                     'SET n:{1}').format(self.task_id, new_label)
            self.graph.run(query, full_name=full_name)
            return True
        return False

    def clear_task_data(self, task_id, batch_size=500):
        """
        Remove a specific label from nodes in batches. If a node only has this one label,
        delete the node.
        """
        remove_label_query_template = """
        MATCH (n:`{label}`)
        WHERE size(labels(n)) > 1
        WITH n LIMIT {limit}
        REMOVE n:`{label}`
        RETURN count(n) AS removed_count
        """

        delete_node_query_template = """
        MATCH (n:`{label}`)
        WHERE size(labels(n)) = 1
        WITH n LIMIT {limit}
        DETACH DELETE n
        RETURN count(n) AS deleted_count
        """

        with self.lock:
            while True:
                remove_label_query = remove_label_query_template.format(
                    label=task_id, limit=batch_size)
                remove_label_result = self.graph.run(remove_label_query).data()
                removed_count = remove_label_result[0]['removed_count']

                delete_node_query = delete_node_query_template.format(
                    label=task_id, limit=batch_size)
                delete_node_result = self.graph.run(delete_node_query).data()
                deleted_count = delete_node_result[0]['deleted_count']

                if removed_count == 0 and deleted_count == 0:
                    break

    def clear_database(self):
        with self.lock:
            self.graph.run('MATCH (n) DETACH DELETE n')

    def execute_query(self, query, **params):
        try:
            with self.lock:
                result = self.graph.run(query, **params)
                return [record for record in result]
        except Exception:
            return ''

    def execute_query_with_exception(self, query, **params):
        try:
            with self.lock:
                result = self.graph.run(query, **params)
                return [record for record in result], True
        except Exception as e:
            return str(e), False

    def execute_query_with_timeout(graph_db, cypher, timeout=60):

        def query_execution():
            return graph_db.execute_query_with_exception(cypher)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(query_execution)
            try:
                cypher_response, flag = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                cypher_response = 'cypher too complex, out of memory'
                flag = True
            except Exception as e:
                cypher_response = str(e)
                flag = False

        return cypher_response, flag

    def update_node(self, full_name, parms={}):
        with self.lock:
            existing_node = self._match_node(full_name)
            if existing_node:
                existing_node.update(parms)
                self.graph.push(existing_node)

    def add_node(self, label, full_name, parms={}):
        with self.lock:
            existing_node = self._match_node(full_name)
            if existing_node:
                if self.none_label in list(existing_node.labels):
                    self._update_node_label(full_name, label)
                elif label not in list(existing_node.labels):
                    self._add_node_label(full_name, label)
                existing_node.update(parms)
                self.graph.push(existing_node)
            else:
                existing_node = self._create_node(
                    label, full_name, parms=parms)
            return existing_node

    def add_edge(
        self,
        start_label=None,
        start_name='',
        relationship_type='',
        end_label=None,
        end_name='',
        params={},
    ):
        with self.lock:
            start_node = self._match_node(full_name=start_name)
            end_node = self._match_node(full_name=end_name)

            if not start_node:
                start_node = self._create_node(
                    start_label, full_name=start_name, parms=params)
            if not end_node:
                end_node = self._create_node(
                    end_label, full_name=end_name, parms=params)

            if start_node and end_node:
                rel = self.rel_matcher.match((start_node, end_node),
                                             relationship_type).first()
                if rel:
                    rel.update(params)
                    self.graph.push(rel)
                    return rel
                else:
                    rel = Relationship(start_node, relationship_type, end_node,
                                       **params)
                    self.graph.create(rel)
                    return rel
            return None

    def update_edge(self,
                    start_name='',
                    relationship_type='',
                    end_name='',
                    params={}):
        with self.lock:
            start_node = self._match_node(full_name=start_name)
            end_node = self._match_node(full_name=end_name)
            if start_node and end_node:
                rel = self.rel_matcher.match((start_node, end_node),
                                             relationship_type).first()
                if rel:
                    rel.update(params)
                    self.graph.push(rel)
                    return rel
                else:
                    rel = Relationship(start_node, relationship_type, end_node,
                                       **params)
                    self.graph.create(rel)
                    return rel
            return None

    def update_file_path(self, root_path):
        with self.lock:
            # 获取所有包含 file_path 属性的节点
            query = (
                'MATCH (n:`{0}`) '
                'WHERE exists(n.file_path)'
                'RETURN n.file_path as file_path, n.full_name as full_name'
            ).format(self.task_id)

            nodes_with_file_path = self.execute_query(query)
            # 遍历每个节点并更新 file_path
            for node in nodes_with_file_path:
                full_name = node['full_name']
                file_path = node['file_path']
                # old_path = node['file_path']
                if file_path.startswith(root_path):
                    file_path = file_path[len(root_path):]
                    self.update_node(
                        full_name=full_name, parms={'file_path': file_path})


class GraphDatabaseHandlerNone:

    def __init__(self, *args, **params):
        pass

    def add_node(self, label, full_name, parms={}):
        pass

    def add_edge(
        self,
        start_label=None,
        start_name='',
        relationship_type='',
        end_label=None,
        end_name='',
        params={},
    ):
        pass
