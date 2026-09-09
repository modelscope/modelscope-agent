from ms_agent.permission.matcher import PermissionMatcher
from ms_agent.permission.suggestions import generate_suggestions


def test_domain_rule_is_exact_and_idna_normalized():
    matcher = PermissionMatcher()
    rule = 'web---fetch:domain:xn--fsqu00a.xn--0zwm56d'

    assert matcher.match_with_content(
        rule, 'web---fetch', {'url': 'https://例子.测试/path'})
    assert not matcher.match_with_content(
        rule, 'web---fetch', {'url': 'https://sub.例子.测试/path'})


def test_subdomains_require_explicit_wildcard():
    matcher = PermissionMatcher()
    rule = 'web---fetch:domain:*.example.com'

    assert matcher.match_with_content(
        rule, 'web---fetch', {'url': 'https://api.example.com/path'})
    assert not matcher.match_with_content(
        rule, 'web---fetch', {'url': 'https://example.com/path'})
    assert not matcher.match_with_content(
        rule, 'web---fetch', {'url': 'https://evil-example.com/path'})


def test_private_loopback_and_credential_urls_are_never_domain_trusted():
    matcher = PermissionMatcher()
    cases = [
        ('domain:localhost', 'http://localhost/admin'),
        ('domain:127.0.0.1', 'http://127.0.0.1/admin'),
        ('domain:10.0.0.1', 'http://10.0.0.1/admin'),
        ('domain:2130706433', 'http://2130706433/admin'),
        ('domain:0x7f000001', 'http://0x7f000001/admin'),
        ('domain:example.com', 'https://user:secret@example.com/admin'),
        ('domain:printer.local', 'http://printer.local/admin'),
        ('domain:svc.internal', 'https://svc.internal/admin'),
        ('domain:metadata.google.internal',
         'http://metadata.google.internal/computeMetadata/v1'),
    ]

    for content_rule, url in cases:
        assert not matcher.match_with_content(
            f'web---fetch:{content_rule}', 'web---fetch', {'url': url})
        assert generate_suggestions('web---fetch', {'url': url}) == []


def test_suggestions_are_ordered_narrow_to_wide():
    shell = generate_suggestions(
        'code_executor---shell_executor', {'command': 'git status'})
    file_rules = generate_suggestions(
        'file_system---read_file', {'path': '/repo/src/main.py'})
    url_rules = generate_suggestions(
        'web---fetch', {'url': 'https://api.example.com/v1?q=1'})
    mcp_rules = generate_suggestions('github---create_issue', {'title': 'x'})

    assert shell[:2] == [
        'code_executor---shell_executor:git status',
        'code_executor---shell_executor:git *',
    ]
    assert file_rules[:2] == [
        'file_system---read_file:/repo/src/main.py',
        'file_system---read_file:/repo/src/*',
    ]
    assert url_rules[:3] == [
        'web---fetch:https://api.example.com/v1[?]q=1',
        'web---fetch:domain:api.example.com',
        'web---fetch',
    ]
    assert mcp_rules == ['github---create_issue', 'github---*']


def test_domain_suggestions_never_auto_widen_to_shared_suffix():
    suggestions = generate_suggestions(
        'web---fetch', {'url': 'https://tenant.github.io/private'})

    assert 'web---fetch:domain:tenant.github.io' in suggestions
    assert not any('domain:*.' in item for item in suggestions)


def test_secret_bearing_or_pipe_literals_are_not_suggested():
    url_rules = generate_suggestions(
        'web---fetch',
        {'url': 'https://example.com/v1?api_key=secret'},
    )
    shell_rules = generate_suggestions(
        'code_executor---shell_executor',
        {'command': 'curl -H "Authorization: Bearer abc" | tee out'},
    )

    assert all('api_key=secret' not in item for item in url_rules)
    assert all('Bearer abc' not in item for item in shell_rules)
    assert all('|' not in item.split(':', 1)[-1] for item in shell_rules)
    assert 'web---fetch:domain:example.com' in url_rules


def test_exact_suggestions_escape_glob_metacharacters():
    matcher = PermissionMatcher()
    suggestions = generate_suggestions(
        'file_system---read_file', {'path': '/repo/file?.txt'})

    assert suggestions[0] == (
        'file_system---read_file:/repo/file[?].txt')
    assert matcher.match_with_content(
        suggestions[0],
        'file_system---read_file',
        {'path': '/repo/file?.txt'},
    )
    assert not matcher.match_with_content(
        suggestions[0],
        'file_system---read_file',
        {'path': '/repo/file1.txt'},
    )

