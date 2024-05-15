import subprocess


def clone_git_repository(repo_url, branch_name, folder_name):
    """
    Clone a git repository into a specified folder.
    Args:
        repo_url: user repository url
        branch_name: branch name
        folder_name: where should the repository be cloned

    Returns:

    """
    try:
        subprocess.run(
            ['git', 'clone', '-b', branch_name, repo_url, folder_name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        print(f"Repository cloned successfully into '{folder_name}'.")

    except subprocess.CalledProcessError as e:
        print(f'Error cloning repository: {e.stderr.decode()}')
        raise RuntimeError(f'Repository cloning failed with e {e}')
