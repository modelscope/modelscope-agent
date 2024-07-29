import codecs
import os
import sys

import jedi
from index_utils import *
from index_utils import _virtualFilePath
from indexer_visitor import AstVisitor, VerboseAstVisitor
from jedi.inference import InferenceState


def isValidEnvironment(environmentPath):
    try:
        environment = jedi.create_environment(environmentPath, False)
        environment._get_subprocess()  # check if this environment is really functional
    except Exception as e:
        if os.name == "nt" and os.path.isdir(environmentPath):
            try:
                environment = jedi.create_environment(
                    os.path.join(environmentPath, "python.exe"), False
                )
                environment._get_subprocess()  # check if this environment is really functional
                return ""
            except Exception:
                pass
        return str(e)
    return ""


def getEnvironment(environmentPath=None):
    if environmentPath is not None:
        try:
            environment = jedi.create_environment(environmentPath, False)
            environment._get_subprocess()  # check if this environment is really functional
            return environment
        except Exception as e:
            if os.name == "nt" and os.path.isdir(environmentPath):
                try:
                    environment = jedi.create_environment(
                        os.path.join(environmentPath, "python.exe"), False
                    )
                    environment._get_subprocess()  # check if this environment is really functional
                    return environment
                except Exception:
                    pass
            print(
                'WARNING: The provided environment path "'
                + environmentPath
                + '" does not specify a functional Python '
                'environment (details: "'
                + str(e)
                + '"). Using fallback environment instead.'
            )

    try:
        environment = jedi.get_default_environment()
        environment._get_subprocess()  # check if this environment is really functional
        return environment
    except Exception:
        pass

    try:
        for environment in jedi.find_system_environments():
            return environment
    except Exception:
        pass

    if (
        os.name == "nt"
    ):  # this is just a workaround and shall be removed once Jedi is fixed (Pull request https://github.com/davidhalter/jedi/pull/1282)
        for version in jedi.api.environment._SUPPORTED_PYTHONS:
            for exe in jedi.api.environment._get_executables_from_windows_registry(
                version
            ):
                try:
                    return jedi.api.environment.Environment(exe)
                except jedi.InvalidPythonEnvironment:
                    pass

    raise jedi.InvalidPythonEnvironment(
        "Unable to find an executable Python environment."
    )


def indexSourceCode(
    sourceCode,
    workingDirectory,
    astVisitorClient,
    isVerbose,
    environmentPath=None,
    sysPath=None,
):
    sourceFilePath = _virtualFilePath

    environment = getEnvironment(environmentPath)

    project = jedi.api.project.Project(
        workingDirectory, environment_path=environment.path
    )

    evaluator = InferenceState(
        project, environment=environment, script_path=workingDirectory
    )

    module_node = evaluator.parse(
        code=sourceCode, path=workingDirectory, cache=False, diff_cache=False
    )

    if isVerbose:
        astVisitor = VerboseAstVisitor(
            astVisitorClient, evaluator, sourceFilePath, sourceCode, sysPath
        )
    else:
        astVisitor = AstVisitor(
            astVisitorClient, evaluator, sourceFilePath, sourceCode, sysPath
        )

    astVisitor.traverseNode(module_node)


def indexSourceFile(
    sourceFilePath,
    environmentPath,
    workingDirectory,
    astVisitorClient,
    isVerbose,
    rootPath,
):

    if isVerbose:
        print('INFO: Indexing source file "' + sourceFilePath + '".')

    sourceCode = ""
    try:
        with codecs.open(sourceFilePath, "r", encoding="utf-8") as input:
            sourceCode = input.read()
    except UnicodeDecodeError:
        print(
            "WARNING: Unable to open source file using utf-8 encoding. Trying to derive encoding automatically."
        )
        with codecs.open(sourceFilePath, "r") as input:
            sourceCode = input.read()

    environment = getEnvironment(environmentPath)

    if isVerbose:
        print(
            'INFO: Using Python environment at "' + environment.path + '" for indexing.'
        )

    project = jedi.api.project.Project(
        workingDirectory, environment_path=environment.path
    )

    evaluator = InferenceState(
        project, environment=environment, script_path=workingDirectory
    )

    module_node = evaluator.parse(
        code=sourceCode, path=workingDirectory, cache=False, diff_cache=False
    )
    astVisitorClient.this_source_code_lines = sourceCode.split("\n")
    if isVerbose:
        astVisitor = VerboseAstVisitor(astVisitorClient, evaluator, sourceFilePath)
    else:
        astVisitor = AstVisitor(
            astVisitorClient, evaluator, sourceFilePath, rootPath=rootPath
        )

    astVisitor.traverseNode(module_node)
