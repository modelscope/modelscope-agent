import codecs
import os
from enum import Enum

import jedi
import parso
import sourcetraildb as srctrl
from index_utils import NameElement, NameHierarchy, SourceRange
from jedi.inference import InferenceState

_virtualFilePath = 'virtual_file.py'


def indexSourceCode(sourceCode,
                    workingDirectory,
                    astVisitorClient,
                    isVerbose,
                    sysPath=None):
    sourceFilePath = _virtualFilePath

    moduleNode = parso.parse(sourceCode)

    if isVerbose:
        astVisitor = VerboseAstVisitor(astVisitorClient, sourceFilePath,
                                       sourceCode, sysPath)
    else:
        astVisitor = AstVisitor(astVisitorClient, sourceFilePath, sourceCode,
                                sysPath)

    astVisitor.traverseNode(moduleNode)


def getEnvironment(environmentPath=None):
    if environmentPath is not None:
        try:
            environment = jedi.create_environment(environmentPath, False)
            environment._get_subprocess(
            )  # check if this environment is really functional
            return environment
        except Exception as e:
            if os.name == 'nt' and os.path.isdir(environmentPath):
                try:
                    environment = jedi.create_environment(
                        os.path.join(environmentPath, 'python.exe'), False)
                    environment._get_subprocess(
                    )  # check if this environment is really functional
                    return environment
                except Exception:
                    pass
            print('WARNING: The provided environment path "' + environmentPath
                  + '" does not specify a functional Python '
                  'environment (details: "' + str(e)
                  + '"). Using fallback environment instead.')

    try:
        environment = jedi.get_default_environment()
        environment._get_subprocess(
        )  # check if this environment is really functional
        return environment
    except Exception:
        pass

    try:
        for environment in jedi.find_system_environments():
            return environment
    except Exception:
        pass

    if (os.name == 'nt'):
        for version in jedi.api.environment._SUPPORTED_PYTHONS:
            for exe in jedi.api.environment._get_executables_from_windows_registry(
                    version):
                try:
                    return jedi.api.environment.Environment(exe)
                except jedi.InvalidPythonEnvironment:
                    pass

    raise jedi.InvalidPythonEnvironment(
        'Unable to find an executable Python environment.')


def indexSourceFile(
    sourceFilePath,
    environmentDirectoryPath,
    workingDirectory,
    astVisitorClient,
    isVerbose,
    rootPath,
):

    if isVerbose:
        print('INFO: Indexing source file "' + sourceFilePath + '".')

    sourceCode = ''
    try:
        with codecs.open(sourceFilePath, 'r', encoding='utf-8') as input:
            sourceCode = input.read()
    except UnicodeDecodeError:
        print(
            'WARNING: Unable to open source file using utf-8 encoding. Trying to derive encoding automatically.'
        )
        with codecs.open(sourceFilePath, 'r') as input:
            sourceCode = input.read()

    environment = getEnvironment(environmentDirectoryPath)

    if isVerbose:
        print('INFO: Using Python environment at "' + environment.path
              + '" for indexing.')

    project = jedi.api.project.Project(
        workingDirectory, environment_path=environment.path)

    evaluator = InferenceState(
        project, environment=environment, script_path=workingDirectory)

    module_node = evaluator.parse(
        code=sourceCode, path=workingDirectory, cache=False, diff_cache=False)
    astVisitorClient.this_source_code_lines = sourceCode.split('\n')
    if isVerbose:
        astVisitor = VerboseAstVisitor(astVisitorClient, evaluator,
                                       sourceFilePath)
    else:
        astVisitor = AstVisitor(
            astVisitorClient, evaluator, sourceFilePath, rootPath=rootPath)

    astVisitor.traverseNode(module_node)


class ContextType(Enum):
    FILE = 1
    MODULE = 2
    CLASS = 3
    FUNCTION = 4
    METHOD = 5


class ContextInfo:

    def __init__(self, id, contextType, name, node):
        self.id = id
        self.name = name
        self.node = node
        self.selfParamName = None
        self.localSymbolNames = []
        self.contextType = contextType


class ReferenceKindInfo:

    def __init__(self, kind, node):
        self.kind = kind
        self.node = node


class AstVisitor:

    def __init__(
        self,
        client,
        evaluator,
        sourceFilePath,
        sourceFileContent=None,
        sysPath=None,
        rootPath=None,
    ):

        self.client = client
        self.environment = evaluator.environment

        self.sourceFilePath = sourceFilePath
        if sourceFilePath != _virtualFilePath:
            self.sourceFilePath = os.path.abspath(self.sourceFilePath)

        self.sourceFileName = os.path.split(self.sourceFilePath)[-1]
        self.sourceFileContent = sourceFileContent

        if rootPath is None:
            packageRootPath = os.path.dirname(self.sourceFilePath)
            while os.path.exists(os.path.join(packageRootPath, '__init__.py')):
                packageRootPath = os.path.dirname(packageRootPath)
        else:
            packageRootPath = rootPath
        self.sysPath = [packageRootPath]

        if sysPath is not None:
            self.sysPath.extend(sysPath)
        else:
            baseSysPath = evaluator.environment.get_sys_path()
            baseSysPath.sort(reverse=True)
            self.sysPath.extend(baseSysPath)
        self.sysPath = list(filter(None, self.sysPath))

        self.contextStack = []
        self.referenceKindStack = []

        fileId = self.client.recordFile(self.sourceFilePath)
        if fileId == 0:
            print('ERROR: ')
        self.client.recordFileLanguage(fileId, 'python')
        self.contextStack.append(
            ContextInfo(fileId, ContextType.FILE, self.sourceFilePath, None))

        moduleNameHierarchy = self.getNameHierarchyFromModuleFilePath(
            self.sourceFilePath)
        if moduleNameHierarchy is not None:
            self.client.this_module = moduleNameHierarchy.getDisplayString()
            moduleId = self.client.recordSymbol(moduleNameHierarchy)
            self.client.recordSymbolDefinitionKind(moduleId,
                                                   srctrl.DEFINITION_EXPLICIT)
            self.client.recordSymbolKind(moduleId, srctrl.SYMBOL_MODULE)
            self.contextStack.append(
                ContextInfo(
                    moduleId,
                    ContextType.MODULE,
                    moduleNameHierarchy.getDisplayString(),
                    None,
                ))

    def traverseNode(self, node):
        if node is None:
            return

        if node.type == 'classdef':
            self.traverseClassdef(node)
        elif node.type == 'funcdef':
            self.traverseFuncdef(node)
        elif node.type == 'param':
            self.traverseParam(node)
        elif node.type == 'argument':
            self.traverseArgument(node)
        elif node.type == 'import_from':
            self.traverseImportFrom(node)
        elif node.type == 'dotted_as_name':
            self.traverseDottedAsNameOrImportAsName(node)
        elif node.type == 'import_as_name':
            self.traverseDottedAsNameOrImportAsName(node)
        else:
            if node.type == 'name':
                self.beginVisitName(node)
            elif node.type == 'string':
                self.beginVisitString(node)
            elif node.type == 'error_leaf':
                self.beginVisitErrorLeaf(node)
            elif node.type == 'import_name':
                self.beginVisitImportName(node)

            if hasattr(node, 'children'):
                for c in node.children:
                    self.traverseNode(c)

            if node.type == 'name':
                self.endVisitName(node)
            elif node.type == 'string':
                self.endVisitString(node)
            elif node.type == 'error_leaf':
                self.endVisitErrorLeaf(node)
            elif node.type == 'import_name':
                self.endVisitImportName(node)

    # ----------------

    def traverseClassdef(self, node):
        if node is None:
            return

        self.beginVisitClassdef(node)

        superArglist = node.get_super_arglist()
        if superArglist is not None:
            self.beginVisitClassdefSuperArglist(superArglist)
            self.traverseNode(superArglist)
            self.endVisitClassdefSuperArglist(superArglist)
        self.traverseNode(node.get_suite())

        self.endVisitClassdef(node)

    def traverseFuncdef(self, node):
        if node is None:
            return

        self.beginVisitFuncdef(node)

        for n in node.get_params():
            self.traverseNode(n)
        self.traverseNode(node.get_suite())

        self.endVisitFuncdef(node)

    def traverseParam(self, node):
        if node is None:
            return

        self.beginVisitParam(node)

        self.traverseNode(node.default)

        self.endVisitParam(node)

    def traverseArgument(self, node):
        if node is None:
            return

        childTraverseStartIndex = 0

        for i in range(len(node.children)):
            if node.children[i].type == 'operator' and node.children[
                    i].value == '=':
                childTraverseStartIndex = i + 1
                break

        for i in range(childTraverseStartIndex, len(node.children)):
            self.traverseNode(node.children[i])

    def traverseImportFrom(self, node):
        if node is None:
            return

        referenceKindAdded = False

        for c in node.children:
            self.traverseNode(c)
            if c.type == 'keyword' and c.value == 'import' and not referenceKindAdded:
                self.referenceKindStack.append(
                    ReferenceKindInfo(srctrl.REFERENCE_IMPORT, node))
                referenceKindAdded = True

        if referenceKindAdded:
            self.referenceKindStack.pop()

    def traverseDottedAsNameOrImportAsName(self, node):
        if node is None:
            return

        referenceKindRemoved = False
        previousReferenceKind = None

        for c in node.children:
            self.traverseNode(c)
            if (c.type == 'keyword' and c.value == 'as'
                    and not referenceKindRemoved
                    and len(self.referenceKindStack) > 0):
                previousReferenceKind = self.referenceKindStack[-1]
                self.referenceKindStack.pop()
                referenceKindRemoved = True

        if referenceKindRemoved:
            self.referenceKindStack.append(previousReferenceKind)

    # ----------------

    def beginVisitClassdef(self, node):
        nameNode = node.name

        symbolNameHierarchy = self.getNameHierarchyOfNode(nameNode)
        if symbolNameHierarchy is None:
            symbolNameHierarchy = getNameHierarchyForUnsolvedSymbol()

        symbolId = self.client.recordSymbol(symbolNameHierarchy)
        self.client.recordSymbolDefinitionKind(symbolId,
                                               srctrl.DEFINITION_EXPLICIT)
        self.client.recordSymbolKind(symbolId, srctrl.SYMBOL_CLASS)
        self.client.recordSymbolLocation(symbolId,
                                         getSourceRangeOfNode(nameNode))
        self.client.recordSymbolScopeLocation(symbolId,
                                              getSourceRangeOfNode(node))
        self.contextStack.append(
            ContextInfo(
                symbolId,
                ContextType.CLASS,
                symbolNameHierarchy.getDisplayString(),
                node,
            ))

    def endVisitClassdef(self, node):
        if len(self.contextStack) > 0:
            contextNode = self.contextStack[-1].node
            if node == contextNode:
                self.contextStack.pop()

    def beginVisitClassdefSuperArglist(self, node):
        self.referenceKindStack.append(
            ReferenceKindInfo(srctrl.REFERENCE_INHERITANCE, node))

    def endVisitClassdefSuperArglist(self, node):
        if len(self.referenceKindStack) > 0:
            referenceKindNode = self.referenceKindStack[-1].node
            if node == referenceKindNode:
                self.referenceKindStack.pop()

    def beginVisitFuncdef(self, node):
        nameNode = node.name

        symbolNameHierarchy = self.getNameHierarchyOfNode(nameNode)
        if symbolNameHierarchy is None:
            symbolNameHierarchy = getNameHierarchyForUnsolvedSymbol()

        selfParamName = None
        localSymbolNames = []

        contextType = ContextType.FUNCTION
        symbolKind = srctrl.SYMBOL_FUNCTION
        if self.contextStack[-1].contextType == ContextType.CLASS:
            contextType = ContextType.METHOD
            symbolKind = srctrl.SYMBOL_METHOD

        for param in node.get_params():
            if contextType == ContextType.METHOD and selfParamName is None:
                selfParamName = param.name.value
            localSymbolNames.append(param.name.value)

        symbolId = self.client.recordSymbol(symbolNameHierarchy)
        self.client.recordSymbolDefinitionKind(symbolId,
                                               srctrl.DEFINITION_EXPLICIT)
        self.client.recordSymbolKind(symbolId, symbolKind)
        self.client.recordSymbolLocation(symbolId,
                                         getSourceRangeOfNode(nameNode))
        self.client.recordSymbolScopeLocation(symbolId,
                                              getSourceRangeOfNode(node))
        contextInfo = ContextInfo(symbolId, contextType,
                                  symbolNameHierarchy.getDisplayString(), node)
        contextInfo.selfParamName = selfParamName
        contextInfo.localSymbolNames.extend(localSymbolNames)
        self.contextStack.append(contextInfo)

    def endVisitFuncdef(self, node):
        if len(self.contextStack) > 0:
            contextNode = self.contextStack[-1].node
            if node == contextNode:
                self.contextStack.pop()

    def beginVisitParam(self, node):
        nameNode = node.name
        localSymbolId = self.client.recordLocalSymbol(
            self.getLocalSymbolName(nameNode))
        self.client.recordLocalSymbolLocation(localSymbolId,
                                              getSourceRangeOfNode(nameNode))

    def endVisitParam(self, node):
        if len(self.contextStack) > 0:
            contextNode = self.contextStack[-1].node
            if node == contextNode:
                self.contextStack.pop()

    def beginVisitName(self, node):
        if len(self.contextStack) == 0:
            return

        if node.value in [
                'True',
                'False',
                'None',
        ]:  # these are not parsed as "keywords" in Python 2
            return

        nextLeafNode = getNextLeaf(node)

        if (nextLeafNode is not None and nextLeafNode.type == 'operator'
                and nextLeafNode.value == '.'):
            symbolNameHierarchy = getNameHierarchyForUnsolvedSymbol()
            symbolId = self.client.recordSymbol(symbolNameHierarchy)
            self.client.recordQualifierLocation(symbolId,
                                                getSourceRangeOfNode(node))
            return

        if len(self.referenceKindStack
               ) > 0 and self.referenceKindStack[-1] is not None:
            if self.referenceKindStack[
                    -1].kind == srctrl.REFERENCE_INHERITANCE:
                self.client.recordReferenceToUnsolvedSymhol(
                    self.contextStack[-1].id,
                    srctrl.REFERENCE_INHERITANCE,
                    getSourceRangeOfNode(node),
                )
                return
            if self.referenceKindStack[-1].kind == srctrl.REFERENCE_IMPORT:
                self.client.recordReferenceToUnsolvedSymhol(
                    self.contextStack[-1].id,
                    srctrl.REFERENCE_IMPORT,
                    getSourceRangeOfNode(node),
                )
                return

        referenceKind = srctrl.REFERENCE_USAGE
        if (nextLeafNode is not None and nextLeafNode.type == 'operator'
                and nextLeafNode.value == '('):
            referenceKind = srctrl.REFERENCE_CALL

        if node.is_definition():
            if referenceKind == srctrl.REFERENCE_CALL:

                pass
            namedDefinitionParentNode = getParentWithTypeInList(
                node, ['classdef', 'funcdef'])
            if namedDefinitionParentNode is not None:
                if namedDefinitionParentNode.type in ['classdef']:
                    if getNamedParentNode(node) == namedDefinitionParentNode:

                        symbolNameHierarchy = self.getNameHierarchyOfNode(node)
                        if symbolNameHierarchy is not None:
                            symbolId = self.client.recordSymbol(
                                symbolNameHierarchy)
                            self.client.recordSymbolKind(
                                symbolId, srctrl.SYMBOL_FIELD)
                            self.client.recordSymbolDefinitionKind(
                                symbolId, srctrl.DEFINITION_EXPLICIT)
                            self.client.recordSymbolLocation(
                                symbolId, getSourceRangeOfNode(node))
                            return
                elif namedDefinitionParentNode.type in ['funcdef']:
                    # definition may be a non-static member variable
                    if (node.parent is not None
                            and node.parent.type == 'trailer'
                            and node.get_previous_sibling() is not None
                            and node.get_previous_sibling().value == '.'):
                        potentialSelfParamNode = getNamedParentNode(node)
                        if (potentialSelfParamNode is not None
                                and getFirstDirectChildWithType(
                                    potentialSelfParamNode, 'name').value
                                == self.contextStack[-1].selfParamName):
                            # definition is a non-static member variable
                            symbolNameHierarchy = self.getNameHierarchyOfNode(
                                node)
                            if symbolNameHierarchy is not None:
                                sourceRange = getSourceRangeOfNode(node)

                                symbolId = self.client.recordSymbol(
                                    symbolNameHierarchy)
                                self.client.recordSymbolKind(
                                    symbolId, srctrl.SYMBOL_FIELD)
                                self.client.recordSymbolDefinitionKind(
                                    symbolId, srctrl.DEFINITION_EXPLICIT)
                                self.client.recordSymbolLocation(
                                    symbolId, sourceRange)

                                referenceId = self.client.recordReference(
                                    self.contextStack[-1].id, symbolId,
                                    referenceKind)
                                self.client.recordReferenceLocation(
                                    referenceId, sourceRange)
                                return
                        else:
                            self.client.recordReferenceToUnsolvedSymhol(
                                self.contextStack[-1].id,
                                referenceKind,
                                getSourceRangeOfNode(node),
                            )
                            return
                    localSymbolId = self.client.recordLocalSymbol(
                        self.getLocalSymbolName(node))
                    self.client.recordLocalSymbolLocation(
                        localSymbolId, getSourceRangeOfNode(node))
                    self.contextStack[-1].localSymbolNames.append(node.value)
                    return
            else:
                symbolNameHierarchy = self.getNameHierarchyOfNode(node)
                if symbolNameHierarchy is not None:
                    symbolId = self.client.recordSymbol(
                        symbolNameHierarchy,
                        global_node=node,
                        node_path=self.sourceFilePath,
                    )
                    self.client.recordSymbolKind(symbolId,
                                                 srctrl.SYMBOL_GLOBAL_VARIABLE)
                    self.client.recordSymbolDefinitionKind(
                        symbolId, srctrl.DEFINITION_EXPLICIT)
                    self.client.recordSymbolLocation(
                        symbolId, getSourceRangeOfNode(node))
                    return
        else:  # if not node.is_definition():
            recordLocalSymbolUsage = True

            if (node.parent is not None and node.parent.type == 'trailer'
                    and node.get_previous_sibling() is not None
                    and node.get_previous_sibling().value == '.'):
                recordLocalSymbolUsage = False

            if recordLocalSymbolUsage:
                if node.value in self.contextStack[-1].localSymbolNames:
                    localSymbolId = self.client.recordLocalSymbol(
                        self.getLocalSymbolName(node))
                    self.client.recordLocalSymbolLocation(
                        localSymbolId, getSourceRangeOfNode(node))
                    return

        # fallback if not returned before
        self.client.recordReferenceToUnsolvedSymhol(self.contextStack[-1].id,
                                                    referenceKind,
                                                    getSourceRangeOfNode(node))

    def endVisitName(self, node):
        if len(self.contextStack) > 0:
            contextNode = self.contextStack[-1].node
            if node == contextNode:
                self.contextStack.pop()

    def beginVisitString(self, node):
        sourceRange = getSourceRangeOfNode(node)
        if sourceRange.startLine != sourceRange.endLine:
            self.client.recordAtomicSourceRange(sourceRange)

    def endVisitString(self, node):
        if len(self.contextStack) > 0:
            contextNode = self.contextStack[-1].node
            if node == contextNode:
                self.contextStack.pop()

    def beginVisitErrorLeaf(self, node):
        self.client.recordError(
            'Unexpected token of type "' + node.token_type + '" encountered.',
            False,
            getSourceRangeOfNode(node),
        )

    def endVisitErrorLeaf(self, node):
        if len(self.contextStack) > 0:
            contextNode = self.contextStack[-1].node
            if node == contextNode:
                self.contextStack.pop()

    def beginVisitImportName(self, node):
        self.referenceKindStack.append(
            ReferenceKindInfo(srctrl.REFERENCE_IMPORT, node))

    def endVisitImportName(self, node):
        if len(self.referenceKindStack) > 0:
            referenceKindNode = self.referenceKindStack[-1].node
            if node == referenceKindNode:
                self.referenceKindStack.pop()

    # ----------------

    def getLocalSymbolName(self, nameNode):
        return str(self.contextStack[-1].name) + '<' + nameNode.value + '>'

    def getNameHierarchyFromModuleFilePath(self, filePath):
        if filePath is None:
            return None

        if filePath == _virtualFilePath:
            return NameHierarchy(
                NameElement(os.path.splitext(_virtualFilePath)[0]), '.')

        filePath = os.path.abspath(filePath)
        # First remove the suffix.
        for suffix in ['.py']:
            if filePath.endswith(suffix):
                filePath = filePath[:-len(suffix)]
                break

        for p in self.sysPath:
            if filePath.startswith(p):
                rest = filePath[len(p):]
                if rest.startswith(os.path.sep):
                    # Remove a slash in cases it's still there.
                    rest = rest[1:]
                if rest:
                    split = rest.split(os.path.sep)
                    for string in split:
                        if not string:
                            return None

                    if split[-1] == '__init__':
                        split = split[:-1]

                    nameHierarchy = None
                    for namePart in split:
                        if nameHierarchy is None:
                            nameHierarchy = NameHierarchy(
                                NameElement(namePart), '.')
                        else:
                            nameHierarchy.nameElements.append(
                                NameElement(namePart))
                    return nameHierarchy

        return None

    def getNameHierarchyOfNode(self, node):
        if node is None:
            return None

        if node.type == 'name':
            nameNode = node
        else:
            nameNode = getFirstDirectChildWithType(node, 'name')

        if nameNode is None:
            return None

        parentNode = getParentWithTypeInList(nameNode.parent,
                                             ['classdef', 'funcdef'])

        if self.contextStack[-1].contextType == ContextType.METHOD:
            potentialSelfNode = getNamedParentNode(node)
            if potentialSelfNode is not None:
                potentialSelfNameNode = getFirstDirectChildWithType(
                    potentialSelfNode, 'name')
                if (potentialSelfNameNode is not None
                        and potentialSelfNameNode.value
                        == self.contextStack[-1].selfParamName):
                    parentNode = self.contextStack[-2].node

        nameElement = NameElement(nameNode.value)

        if parentNode is not None:
            parentNodeNameHierarchy = self.getNameHierarchyOfNode(parentNode)
            if parentNodeNameHierarchy is None:
                return None
            parentNodeNameHierarchy.nameElements.append(nameElement)
            return parentNodeNameHierarchy

        nameHierarchy = self.getNameHierarchyFromModuleFilePath(
            self.sourceFilePath)
        if nameHierarchy is None:
            return None
        nameHierarchy.nameElements.append(nameElement)
        return nameHierarchy

        return None


class VerboseAstVisitor(AstVisitor):

    def __init__(self,
                 client,
                 sourceFilePath,
                 sourceFileContent=None,
                 sysPath=None):
        AstVisitor.__init__(self, client, sourceFilePath, sourceFileContent,
                            sysPath)
        self.indentationLevel = 0
        self.indentationToken = '| '

    def traverseNode(self, node):
        if node is None:
            return

        currentString = ''
        for i in range(0, self.indentationLevel):
            currentString += self.indentationToken

        currentString += node.type

        if hasattr(node, 'value'):
            currentString += ' (' + repr(node.value) + ')'

        currentString += ' ' + getSourceRangeOfNode(node).toString()

        print('AST: ' + currentString)

        self.indentationLevel += 1
        AstVisitor.traverseNode(self, node)
        self.indentationLevel -= 1


def getNameHierarchyForUnsolvedSymbol():
    return NameHierarchy(NameElement(NameHierarchy.unsolvedSymbolName), '')


def isQualifierNode(node):
    nextNode = getNext(node)
    if nextNode is not None and nextNode.type == 'trailer':
        nextNode = getNext(nextNode)
    if nextNode is not None and nextNode.type == 'operator' and nextNode.value == '.':
        return True
    return False


def getSourceRangeOfNode(node):
    startLine, startColumn = node.start_pos
    endLine, endColumn = node.end_pos
    return SourceRange(startLine, startColumn + 1, endLine, endColumn)


def getNamedParentNode(node):
    if node is None:
        return None

    parentNode = node.parent

    if node.type == 'name' and parentNode is not None:
        parentNode = parentNode.parent

    while parentNode is not None:
        if getFirstDirectChildWithType(parentNode, 'name') is not None:
            return parentNode
        parentNode = parentNode.parent

    return None


def getParentWithType(node, type):
    if node is None:
        return None
    parentNode = node.parent
    if parentNode is None:
        return None
    if parentNode.type == type:
        return parentNode
    return getParentWithType(parentNode, type)


def getParentWithTypeInList(node, typeList):
    if node is None:
        return None
    parentNode = node.parent
    if parentNode is None:
        return None
    if parentNode.type in typeList:
        return parentNode
    return getParentWithTypeInList(parentNode, typeList)


def getFirstDirectChildWithType(node, type):
    for c in node.children:
        if c.type == type:
            return c
    return None


def getDirectChildrenWithType(node, type):
    children = []
    for c in node.children:
        if c.type == type:
            children.append(c)
    return children


def getNext(node):
    if hasattr(node, 'children'):
        for c in node.children:
            return c

    siblingSource = node
    while siblingSource is not None and siblingSource.parent is not None:
        sibling = siblingSource.get_next_sibling()
        if sibling is not None:
            return sibling
        siblingSource = siblingSource.parent

    return None


def getNextLeaf(node):
    nextNode = getNext(node)
    while nextNode is not None and hasattr(nextNode, 'children'):
        nextNode = getNext(nextNode)
    return nextNode
