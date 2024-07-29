import os
import time
from functools import wraps


def method_decorator(func):
    """
    装饰器：在调用函数之前和之后打印消息，并正确处理类方法中的`self`参数。
    """

    def wrapped(self, *args, **kwargs):
        # start_time = time.time()
        result = func(self, *args, **kwargs)
        # end_time = time.time()
        # print(
        #     f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds"
        # )
        return result

    return wrapped


class TimerDecorator:

    def __init__(self, func):
        self.func = func
        wraps(func)(self)

    def __call__(self, *args, **kwargs):

        result = self.func(*args, **kwargs)
        return result


def get_py_files(directory):
    py_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files


def convert_to_relative_path(root_path, working_path):
    return os.path.relpath(root_path, working_path)


def convert_dotted_name(root_path, file_path):

    def get_dotted_name(path):
        rest = path[len(root_path):]
        split = [s for s in rest.split(os.path.sep) if s != '']
        return '.'.join(split)

    file_folder_path = os.path.dirname(file_path)
    file_path = file_path.split('.py')[0]
    if file_folder_path.startswith(root_path):
        parent_module_path = get_dotted_name(file_folder_path)
        module_path = get_dotted_name(file_path)
        return parent_module_path, module_path
    else:
        return None, None


def get_dotted_name(root_path, file_path):

    def _get_dotted_name(path):
        rest = path[len(root_path):]
        split = [s for s in rest.split(os.path.sep) if s != '']
        return '.'.join(split)

    if '.py' in file_path:
        return _get_dotted_name(file_path.split('.py')[0])
    else:
        return _get_dotted_name(file_path)


def get_module_name(file_path, node, working_path):

    file_dir_path = os.path.dirname(file_path)

    def get_dotted_name(path):
        rest = path[len(working_path):]
        split = [s for s in rest.split(os.path.sep) if s != '']
        return '.'.join(split)

    node_path = str(node.module).replace('.', os.path.sep)

    # print('is ', os.path.join(working_path, node_path + '.py'))
    if os.path.exists(os.path.join(working_path, node_path + '.py')):
        # print('is ', os.path.join(working_path, node_path + '.py'), node.module)
        return node.module
    if os.path.exists(os.path.join(working_path, node_path, '__init__.py')):
        # print('is ', os.path.join(working_path, node_path + '.py'), node.module)
        return node.module

    # Construct the relative path
    if node.level > 0:
        if node.module:
            relative_path = f"{'.' * node.level}{os.path.sep}{node.module}"
        else:
            relative_path = f"{'.' * node.level}{os.path.sep}"
    else:
        relative_path = node.module if node.module else ''
    # Get the absolute path of the module
    current_file_path = os.path.abspath(file_dir_path)
    absolute_path = os.path.abspath(
        os.path.join(current_file_path, relative_path))

    if absolute_path.startswith(working_path):
        module_path = get_dotted_name(absolute_path)
        return module_path
    else:
        return None

    # Calculate the relative path to the working path
    # relative_to_working_path = os.path.relpath(absolute_path, )
    # Convert file system path to module path
    # module_path = relative_to_working_path.replace(os.sep, '.')


def module_name_to_path(module_path, working_path):
    # Convert module path to file path
    relative_file_path = module_path.replace('.', os.sep)
    absolute_file_path = os.path.join(working_path, relative_file_path)

    # Normalize the path to handle any redundant separators
    absolute_file_path = os.path.normpath(absolute_file_path)

    return absolute_file_path


if __name__ == '__main__':
    repo_path = r'/home/lanbo/repo/test_repo'
    py_files = get_py_files(repo_path)
    # print(py_files)
    print(
        convert_dotted_name(repo_path,
                            r'/home/lanbo/repo/test_repo/file1/file2.py')[1])
    print(
        module_name_to_path(
            convert_dotted_name(
                repo_path, r'/home/lanbo/repo/test_repo/file1/file2.py')[1],
            repo_path,
        ))
