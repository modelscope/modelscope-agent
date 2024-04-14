from setuptools import find_packages, setup
from typing import List


def parse_requirements(file_name: str) -> List[str]:
    with open(file_name) as f:
        return [
            require.strip() for require in f
            if require.strip() and not require.startswith('#')
        ]


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'modelscope_agent/version.py'


def get_version():
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


setup(
    name=
    'modelscope-agent',  # Replace 'your_package_name' with the name of your package
    version=get_version(),  # Replace with the desired version number
    description=
    'ModelScope Agent: Be a powerful models and tools agent based on ModelScope and open source LLM.',
    author='Modelscope Team',
    author_email='contact@modelscope.cn',
    keywords='python,agent,LLM,AIGC,qwen,ModelScope',
    url=
    'https://github.com/modelscope/modelscope-agent',  # Replace with your repository URL
    license='Apache License 2.0',
    packages=find_packages(exclude=['*test*', 'demo']),
    include_package_data=True,
    install_requires=parse_requirements('requirements.txt'),
    long_description=readme(),
    long_description_content_type='text/markdown',
)
