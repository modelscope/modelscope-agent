import os
import shutil
import tempfile

import pytest
from modelscope_agent.utils.git import clone_git_repository


@pytest.fixture(scope='function')
def temp_dir():
    # create temp dir
    temp_directory = tempfile.mkdtemp()
    yield temp_directory
    # delete temp dir after test
    shutil.rmtree(temp_directory)


def test_clone_git_repository_success(temp_dir):
    # use temp dir as folder name
    repo_url = 'http://www.modelscope.cn/studios/zhicheng/zzc_tool_test.git'
    branch_name = 'master'
    folder_name = temp_dir

    # store the git to local dir
    clone_git_repository(repo_url, branch_name, folder_name)

    # check if success
    assert os.listdir(
        folder_name) != [], 'Directory should not be empty after cloning'


def test_clone_git_repository_failed(temp_dir):
    # use temp dir as folder name
    repo_url = 'http://www.modelscope.cn/studios/zhicheng/zzc_tool_test1.git'
    branch_name = 'master'
    folder_name = temp_dir

    # store the git to local dir
    with pytest.raises(RuntimeError):
        clone_git_repository(repo_url, branch_name, folder_name)

    # check if error
    assert os.listdir(
        folder_name) == [], 'Directory should not be empty after cloning'
