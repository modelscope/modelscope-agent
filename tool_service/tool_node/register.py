import ast
import time
from typing import Dict, List, Optional, Union

# 假设你的Python代码在一个文件中，这里需要加载文件内容
filename = '/Users/zhicheng/repo/maas-agent/modelscope_agent/tools/amap_weather.py'
with open(filename, 'r') as file:
    code = file.read()

# 解析字符串形式的代码为AST
parsed_code = ast.parse(code)


class AttributeExtractor(ast.NodeVisitor):

    def __init__(self, attributes):
        self.attributes = attributes
        self.classes_with_attributes = {}

    def visit_ClassDef(self, node):
        # 用于存放当前类的属性
        found_attributes = {}
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target,
                                  ast.Name) and target.id in self.attributes:
                        found_attributes[target.id] = self.get_value(
                            item.value)
            elif isinstance(item, ast.AnnAssign) and isinstance(
                    item.target, ast.Name):
                if item.target.id in self.attributes:
                    found_attributes[item.target.id] = self.get_value(
                        item.value)

        # 如果找到任何指定的属性，记录这个类
        if found_attributes:
            self.classes_with_attributes[node.name] = found_attributes

    def get_value(self, node):
        """处理不同类型的节点，提取其值"""
        if isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.List):
            return [self.get_value(elem) for elem in node.elts]
        elif isinstance(node, ast.Dict):
            return dict((self.get_value(k), self.get_value(v))
                        for k, v in zip(node.keys, node.values))
        # 可以根据需要添加更多类型处理
        return None


s1 = time.time()

# 创建访问者实例，指定我们要查找的属性
extractor = AttributeExtractor(['name', 'parameters', 'description'])

# 使用访问者遍历AST
extractor.visit(parsed_code)

# 输出提取的属性
print(extractor.classes_with_attributes)
print(time.time() - s1)
