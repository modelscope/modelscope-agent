import ast
import inspect
import time


class AttributeExtractor(ast.NodeVisitor):

    def __init__(self, attributes):
        self.attributes = attributes
        self.classes_with_attributes = {}

    def visit_ClassDef(self, node):
        # store the attributes of the class
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

        # if found any attributes, store them
        if found_attributes:
            self.classes_with_attributes[node.name] = found_attributes

    def get_value(self, node):
        """Get the value of a node from different types."""
        if isinstance(node, ast.Constant):
            return node.s
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.List):
            return [self.get_value(elem) for elem in node.elts]
        elif isinstance(node, ast.Dict):
            return dict((self.get_value(k), self.get_value(v))
                        for k, v in zip(node.keys, node.values))
        # add more types if needed
        return None


def get_attribute_from_tool_cls(tool_cls):
    # get string from class object
    source_code = inspect.getsource(tool_cls)

    # parse code string into ast
    parsed_code = ast.parse(source_code)

    extractor = AttributeExtractor(['name', 'parameters', 'description'])

    # Traverse the ast
    extractor.visit(parsed_code)

    return extractor.classes_with_attributes


if __name__ == '__main__':
    from modelscope_agent.tools import BaseTool
    from modelscope_agent.tools.modelscope_tools.text_to_video_tool import TextToVideoTool
    s1 = time.time()

    test = get_attribute_from_tool_cls(TextToVideoTool)
    print(test)
    print(type(test))
    print(time.time() - s1)
