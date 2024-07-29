import os
import subprocess
import sys


def main():
    # Determine the project root directory
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..'))
    # Add project root to PYTHONPATH if not already in it
    if project_root not in sys.path:
        sys.path.append(project_root)

    # Export PYTHONPATH for subprocess
    os.environ['PYTHONPATH'] = project_root

    # Path to your Streamlit app
    streamlit_app_path = os.path.join(project_root, 'apps', 'codexgraph_agent',
                                      'help.py')

    # Print PYTHONPATH for debugging purposes
    print(f"PYTHONPATH is set to: {os.environ['PYTHONPATH']}")

    # Run the Streamlit app
    subprocess.run(['streamlit', 'run', streamlit_app_path])


if __name__ == '__main__':
    main()
