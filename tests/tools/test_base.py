import pytest


def add(a, b):
    return a + b


def test_add_function(mocker):
    # 使用 mocker.patch() 模拟函数
    mocker.patch('test_base.add', return_value=10)

    result = add(2, 3)

    # 断言调用次数
    assert add.call_count == 1

    # 断言参数
    add.assert_called_once_with(2, 3)

    # 断言返回值
    assert result == 10
