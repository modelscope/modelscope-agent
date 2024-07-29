import os

import sourcetraildb as srctrl
from index_utils import *
from index_utils import _virtualFilePath


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
            while os.path.exists(os.path.join(packageRootPath, "__init__.py")):
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

        self.client.this_script = self.createScript(self.sourceFilePath)

        fileId = self.client.recordFile(self.sourceFilePath)
        if fileId == 0:
            print("ERROR: " + srctrl.getLastError())
        self.client.recordFileLanguage(fileId, "python")
        self.contextStack.append(ContextInfo(fileId, self.sourceFilePath, None))
        moduleNameHierarchy = self.getNameHierarchyFromModuleFilePath(
            self.sourceFilePath
        )
        if moduleNameHierarchy is not None:
            self.client.this_module = moduleNameHierarchy.getDisplayString()
            moduleId = self.client.recordSymbol(
                moduleNameHierarchy, self.sourceFilePath
            )
            self.client.recordSymbolDefinitionKind(moduleId, srctrl.DEFINITION_EXPLICIT)
            self.client.recordSymbolKind(moduleId, srctrl.SYMBOL_MODULE)
            self.contextStack.append(
                ContextInfo(moduleId, moduleNameHierarchy.getDisplayString(), None)
            )

    def traverseNode(self, node):
        if node is None:
            return
        if node.type == "classdef":
            self.beginVisitClassdef(node)
        elif node.type == "funcdef":
            self.beginVisitFuncdef(node)
        if node.type == "import_from":
            self.beginVisitImportFrom(node)
        if node.type == "import_name":
            self.beginVisitImportName(node)
        elif node.type == "name":
            self.beginVisitName(node)
        elif node.type == "string":
            self.beginVisitString(node)
        elif node.type == "error_leaf":
            self.beginVisitErrorLeaf(node)

        if hasattr(node, "children"):
            for c in node.children:
                self.traverseNode(c)

        if node.type == "classdef":
            self.endVisitClassdef(node)
        elif node.type == "funcdef":
            self.endVisitFuncdef(node)
        if node.type == "import_from":
            self.endVisitImportFrom(node)
        if node.type == "import_name":
            self.endVisitImportName(node)
        elif node.type == "name":
            self.endVisitName(node)
        elif node.type == "string":
            self.endVisitString(node)
        elif node.type == "error_leaf":
            self.endVisitErrorLeaf(node)

    def beginVisitClassdef(self, node):
        nameNode = getFirstDirectChildWithType(node, "name")
        symbolNameHierarchy = self.getNameHierarchyOfNode(nameNode, self.sourceFilePath)
        if symbolNameHierarchy is None:
            symbolNameHierarchy = getNameHierarchyForUnsolvedSymbol()

        symbolId = self.client.recordSymbol(symbolNameHierarchy, self.sourceFilePath)
        self.client.recordSymbolDefinitionKind(symbolId, srctrl.DEFINITION_EXPLICIT)
        self.client.recordSymbolKind(symbolId, srctrl.SYMBOL_CLASS)
        self.client.recordSymbolLocation(symbolId, getSourceRangeOfNode(nameNode))
        self.client.recordSymbolScopeLocation(symbolId, getSourceRangeOfNode(node))
        self.contextStack.append(
            ContextInfo(symbolId, symbolNameHierarchy.getDisplayString(), node)
        )

    def endVisitClassdef(self, node):
        if len(self.contextStack) > 0:
            contextNode = self.contextStack[-1].node
            if node == contextNode:
                self.contextStack.pop()

    def beginVisitFuncdef(self, node):
        nameNode = getFirstDirectChildWithType(node, "name")

        symbolNameHierarchy = self.getNameHierarchyOfNode(nameNode, self.sourceFilePath)
        if symbolNameHierarchy is None:
            symbolNameHierarchy = getNameHierarchyForUnsolvedSymbol()

        symbolId = self.client.recordSymbol(symbolNameHierarchy, self.sourceFilePath)
        self.client.recordSymbolDefinitionKind(symbolId, srctrl.DEFINITION_EXPLICIT)
        self.client.recordSymbolKind(symbolId, srctrl.SYMBOL_FUNCTION)
        self.client.recordSymbolLocation(symbolId, getSourceRangeOfNode(nameNode))
        self.client.recordSymbolScopeLocation(symbolId, getSourceRangeOfNode(node))
        self.contextStack.append(
            ContextInfo(symbolId, symbolNameHierarchy.getDisplayString(), node)
        )

        self.recordFunctionOverrideEdge(nameNode)

    def recordFunctionOverrideEdge(self, functionNameNode):
        try:
            functionNameHierarchy = self.getNameHierarchyOfNode(
                functionNameNode, self.sourceFilePath
            )
            if functionNameHierarchy is None:
                return
            functionSymbolId = self.client.recordSymbol(
                functionNameHierarchy, self.sourceFilePath
            )

            (startLine, startColumn) = functionNameNode.start_pos
            script = self.createScript(self.sourceFilePath)
            for definition in script.goto(
                line=startLine,
                column=startColumn,
                follow_imports=True,
                follow_override=True,
            ):
                if definition is None:
                    continue

                overriddenNameNode = definition._name.tree_name

                if functionNameNode.start_pos == overriddenNameNode.start_pos:
                    continue

                overriddenNameHierarchy = self.getNameHierarchyOfNode(
                    overriddenNameNode, self.sourceFilePath
                )
                if overriddenNameHierarchy is None:
                    continue
                overriddenSymbolId = self.client.recordSymbol(
                    overriddenNameHierarchy, self.sourceFilePath
                )

                referenceId = self.client.recordReference(
                    functionSymbolId, overriddenSymbolId, srctrl.REFERENCE_OVERRIDE
                )

                self.client.recordReferenceLocation(
                    referenceId, getSourceRangeOfNode(overriddenNameNode)
                )
        except Exception:
            pass

    def endVisitFuncdef(self, node):
        if len(self.contextStack) > 0:
            contextNode = self.contextStack[-1].node
            if node == contextNode:
                self.contextStack.pop()

    def beginVisitImportName(self, node):
        self.recordErrorsForUnsolvedImports(node)

    def endVisitImportName(self, node):
        if len(self.contextStack) > 0:
            contextNode = self.contextStack[-1].node
            if node == contextNode:
                self.contextStack.pop()

    def beginVisitImportFrom(self, node):
        self.recordErrorsForUnsolvedImports(node)

    def endVisitImportFrom(self, node):
        if len(self.contextStack) > 0:
            contextNode = self.contextStack[-1].node
            if node == contextNode:
                self.contextStack.pop()

    def beginVisitName(self, node):
        if len(self.contextStack) == 0:
            return

        if node.value in [
            "True",
            "False",
            "None",
        ]:  # these are not parsed as "keywords" in Python 2
            return

        referenceIsUnsolved = True
        for definition in self.getDefinitionsOfNode(node, self.sourceFilePath):
            if definition is None:
                continue

            try:
                if definition.type == "instance":
                    if definition.line is None and definition.column is None:
                        if self.recordInstanceReference(node, definition):
                            referenceIsUnsolved = False

                elif definition.type == "module":
                    if self.recordModuleReference(node, definition):
                        referenceIsUnsolved = False

                elif definition.type in ["class", "function"]:
                    (startLine, startColumn) = node.start_pos
                    if (
                        definition.line == startLine
                        and definition.column == startColumn
                        and definition.module_path == self.sourceFilePath
                    ):
                        # Early exit. We don't record references for locations of classes or functions that are definitions
                        return

                    if definition.type == "class":
                        if self.recordClassReference(node, definition):
                            referenceIsUnsolved = False

                    elif definition.type == "function":
                        if self.recordFunctionReference(node, definition):
                            referenceIsUnsolved = False

                elif definition.type == "param":
                    if definition.line is None or definition.column is None:
                        # Early skip and try next definition. For now we don't record references for names that don't have a valid definition location
                        continue

                    if self.recordParamReference(node, definition):
                        referenceIsUnsolved = False

                elif definition.type == "statement":
                    if definition.line is None or definition.column is None:
                        # Early skip and try next definition. For now we don't record references for names that don't have a valid definition location
                        continue

                    if self.recordStatementReference(node, definition):
                        referenceIsUnsolved = False
            except Exception as e:
                print(
                    'ERROR: Encountered exception "'
                    + e.__repr__()
                    + '" while trying to solve the definition of node "'
                    + node.value
                    + '" at '
                    + getSourceRangeOfNode(node).toString()
                    + "."
                )

        if referenceIsUnsolved:
            self.client.recordReferenceToUnsolvedSymhol(
                self.contextStack[-1].id,
                srctrl.REFERENCE_USAGE,
                getSourceRangeOfNode(node),
            )

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

    def recordErrorsForUnsolvedImports(self, node):
        """
        对于import as name 可以自动转换name成import的类别
        """
        if node.type == "import_from":
            for c in node.children:
                if self.recordErrorsForUnsolvedImports(c) is False:
                    return False
        elif node.type == "import_as_names":
            for c in node.children:
                self.recordErrorsForUnsolvedImports(c)
        elif node.type == "import_as_name":
            for c in node.children:
                if (
                    c.type == "keyword"
                ):  # we just the children (usually only one) until we hit the "as" keyword
                    break
                self.recordErrorsForUnsolvedImports(c)
        elif node.type == "import_name":
            for c in node.children:
                self.recordErrorsForUnsolvedImports(c)
        elif node.type == "dotted_as_names":
            for c in node.children:
                self.recordErrorsForUnsolvedImports(c)
        elif node.type == "dotted_as_name":
            for c in node.children:
                if (
                    c.type == "keyword"
                ):  # we just the children (usually only one) until we hit the "as" keyword
                    break
                self.recordErrorsForUnsolvedImports(c)
        elif node.type == "dotted_name":
            for c in node.children:
                if self.recordErrorsForUnsolvedImports(c) is False:
                    return False
        elif node.type == "name":
            if len(self.getDefinitionsOfNode(node, self.sourceFilePath)) == 0:
                self.client.recordError(
                    'Imported symbol named "' + node.value + '" has not been found.',
                    False,
                    getSourceRangeOfNode(node),
                )
                return False
        return True

    def recordInstanceReference(self, node, definition):
        nameHierarchy = self.getNameHierarchyFromFullNameOfDefinition(definition)
        if nameHierarchy is not None:
            referencedSymbolId = self.client.recordSymbol(
                nameHierarchy, node_path=definition.module_path
            )
            self.client.recordSymbolKind(
                referencedSymbolId, srctrl.SYMBOL_GLOBAL_VARIABLE
            )

            referenceKind = srctrl.REFERENCE_USAGE
            if getParentWithType(node, "import_from") is not None:
                # this would be the case for "from foo import f as my_f"
                #                                             ^    ^
                referenceKind = srctrl.REFERENCE_IMPORT

            referenceId = self.client.recordReference(
                self.contextStack[-1].id, referencedSymbolId, referenceKind
            )
            self.client.recordReferenceLocation(referenceId, getSourceRangeOfNode(node))
            return True
        return False

    def recordModuleReference(self, node, definition):
        referencedNameHierarchy, module_path = (
            self.getNameHierarchyFromModulePathOfDefinition(definition)
        )
        if referencedNameHierarchy is None:
            referencedNameHierarchy = self.getNameHierarchyFromFullNameOfDefinition(
                definition
            )
        if referencedNameHierarchy is None:
            return False

        referencedSymbolId = self.client.recordSymbol(
            referencedNameHierarchy, node_path=module_path
        )

        # Record symbol kind. If the used type is within indexed code, we already have this info. In any other case, this is valuable info!
        self.client.recordSymbolKind(referencedSymbolId, srctrl.SYMBOL_MODULE)

        if isQualifierNode(node):
            self.client.recordQualifierLocation(
                referencedSymbolId, getSourceRangeOfNode(node)
            )
        else:
            referenceKind = srctrl.REFERENCE_USAGE
            if getParentWithType(node, "import_name") is not None:
                # this would be the case for "import foo"
                #                                    ^
                referenceKind = srctrl.REFERENCE_IMPORT

            referenceId = self.client.recordReference(
                self.contextStack[-1].id, referencedSymbolId, referenceKind
            )

            self.client.recordReferenceLocation(referenceId, getSourceRangeOfNode(node))
        return True

    def recordClassReference(self, node, definition):
        referencedNameHierarchy, node_path, tree_node = (
            self.getNameHierarchyOfClassOrFunctionDefinition(definition)
        )
        if referencedNameHierarchy is None:
            return False

        referencedSymbolId = self.client.recordSymbol(
            referencedNameHierarchy, node_path=node_path, tree_node=tree_node
        )

        # Record symbol kind. If the used type is within indexed code, we already have this info. In any other case, this is valuable info!
        self.client.recordSymbolKind(referencedSymbolId, srctrl.SYMBOL_CLASS)

        if isQualifierNode(node):
            self.client.recordQualifierLocation(
                referencedSymbolId, getSourceRangeOfNode(node)
            )
        else:
            referenceKind = srctrl.REFERENCE_TYPE_USAGE
            if node.parent is not None:
                if node.parent.type == "classdef":
                    # this would be the case for "class Foo(Bar)"
                    #                                       ^
                    referenceKind = srctrl.REFERENCE_INHERITANCE
                elif (
                    node.parent.type in ["arglist", "testlist"]
                    and node.parent.parent is not None
                    and node.parent.parent.type == "classdef"
                ):
                    # this would be the case for "class Foo(Bar, Baz)"
                    #                                       ^    ^
                    referenceKind = srctrl.REFERENCE_INHERITANCE
                elif getParentWithType(node, "import_from") is not None:
                    # this would be the case for "from foo import Foo as F"
                    #                                             ^      ^
                    referenceKind = srctrl.REFERENCE_IMPORT

            referenceId = self.client.recordReference(
                self.contextStack[-1].id, referencedSymbolId, referenceKind
            )
            self.client.recordReferenceLocation(referenceId, getSourceRangeOfNode(node))

            if referenceKind == srctrl.REFERENCE_TYPE_USAGE and isCallNode(node):
                constructorNameHierarchy = referencedNameHierarchy.copy()
                constructorNameHierarchy.nameElements.append(NameElement("__init__"))
                constructorSymbolId = self.client.recordSymbol(
                    constructorNameHierarchy, node_path=node_path, tree_node=tree_node
                )
                self.client.recordSymbolKind(constructorSymbolId, srctrl.SYMBOL_METHOD)
                callReferenceId = self.client.recordReference(
                    self.contextStack[-1].id, constructorSymbolId, srctrl.REFERENCE_CALL
                )
                self.client.recordReferenceLocation(
                    callReferenceId, getSourceRangeOfNode(node)
                )

        return True

    def recordFunctionReference(self, node, definition):
        referencedNameHierarchy, node_path, tree_node = (
            self.getNameHierarchyOfClassOrFunctionDefinition(definition)
        )
        if referencedNameHierarchy is None:
            return False

        referencedSymbolId = self.client.recordSymbol(
            referencedNameHierarchy, node_path=node_path, tree_node=tree_node
        )

        # Record symbol kind. If the called function is within indexed code, we already have this info. In any other case, this is valuable info!
        self.client.recordSymbolKind(referencedSymbolId, srctrl.SYMBOL_FUNCTION)

        referenceKind = -1
        if isCallNode(node):
            referenceKind = srctrl.REFERENCE_CALL
        elif getParentWithType(node, "import_from"):
            referenceKind = srctrl.REFERENCE_IMPORT

        if referenceKind == -1:
            return False

        referenceId = self.client.recordReference(
            self.contextStack[-1].id, referencedSymbolId, referenceKind
        )

        self.client.recordReferenceLocation(referenceId, getSourceRangeOfNode(node))
        return True

    def recordParamReference(self, node, definition):
        localSymbolId = self.client.recordLocalSymbol(
            self.getLocalSymbolName(definition)
        )
        self.client.recordLocalSymbolLocation(localSymbolId, getSourceRangeOfNode(node))
        return True

    def recordStatementReference(self, node, definition):
        definitionModulePath = definition.module_path
        if definitionModulePath is None:
            if self.sourceFilePath == _virtualFilePath:
                definitionModulePath = self.sourceFilePath
            else:
                return False

        symbolKind = None
        referenceKind = None
        definitionKind = None

        definitionNameNode = definition._name.tree_name
        namedDefinitionParentNode = getParentWithTypeInList(
            definitionNameNode, ["classdef", "funcdef"]
        )
        if namedDefinitionParentNode is not None:
            if namedDefinitionParentNode.type in ["classdef"]:
                if getNamedParentNode(definitionNameNode) == namedDefinitionParentNode:
                    # definition is not local to some other field instantiation but instead it is a static member variable
                    if (
                        definitionNameNode.start_pos == node.start_pos
                        and definitionNameNode.end_pos == node.end_pos
                    ):
                        # node is the definition of the static member variable
                        symbolKind = srctrl.SYMBOL_FIELD
                        definitionKind = srctrl.DEFINITION_EXPLICIT
                    else:
                        # node is a usage of the static member variable
                        referenceKind = srctrl.REFERENCE_USAGE
            elif namedDefinitionParentNode.type in ["funcdef"]:
                # definition may be a non-static member variable
                if (
                    definitionNameNode.parent is not None
                    and definitionNameNode.parent.type == "trailer"
                ):
                    potentialParamNode = getNamedParentNode(definitionNameNode)
                    if potentialParamNode is not None:
                        for potentialParamDefinition in self.getDefinitionsOfNode(
                            potentialParamNode, definitionModulePath
                        ):
                            if (
                                potentialParamDefinition is not None
                                and potentialParamDefinition.type == "param"
                            ):
                                paramDefinitionNameNode = (
                                    potentialParamDefinition._name.tree_name
                                )
                                potentialFuncdefNode = getNamedParentNode(
                                    paramDefinitionNameNode
                                )
                                if (
                                    potentialFuncdefNode is not None
                                    and potentialFuncdefNode.type == "funcdef"
                                ):
                                    potentialClassdefNode = getNamedParentNode(
                                        potentialFuncdefNode
                                    )
                                    if (
                                        potentialClassdefNode is not None
                                        and potentialClassdefNode.type == "classdef"
                                    ):
                                        preceedingNode = (
                                            paramDefinitionNameNode.parent.get_previous_sibling()
                                        )
                                        if (
                                            preceedingNode is not None
                                            and preceedingNode.type != "param"
                                        ):
                                            # 'paramDefinitionNameNode' is the first parameter of a member function (aka. 'self')
                                            referenceKind = srctrl.REFERENCE_USAGE
                                            if (
                                                definitionNameNode.start_pos
                                                == node.start_pos
                                                and definitionNameNode.end_pos
                                                == node.end_pos
                                            ):
                                                symbolKind = srctrl.SYMBOL_FIELD
                                                definitionKind = (
                                                    srctrl.DEFINITION_EXPLICIT
                                                )
        else:
            symbolKind = srctrl.SYMBOL_GLOBAL_VARIABLE
            if (
                definitionNameNode.start_pos == node.start_pos
                and definitionNameNode.end_pos == node.end_pos
            ):
                # node is the definition of a global variable
                definitionKind = srctrl.DEFINITION_EXPLICIT
            elif getParentWithType(node, "import_from") is not None:
                # this would be the case for "from foo import f as my_f"
                #                                             ^    ^
                referenceKind = srctrl.REFERENCE_IMPORT
            else:
                referenceKind = srctrl.REFERENCE_USAGE

        sourceRange = getSourceRangeOfNode(node)

        if symbolKind is not None or referenceKind is not None:
            symbolNameHierarchy = self.getNameHierarchyOfNode(
                definitionNameNode, definitionModulePath
            )
            if symbolNameHierarchy is None:
                return False

            symbolId = self.client.recordSymbol(
                symbolNameHierarchy, node_path=definitionModulePath
            )

            if symbolKind is not None:
                self.client.recordSymbolKind(symbolId, symbolKind)

            if definitionKind is not None:
                self.client.recordSymbolDefinitionKind(symbolId, definitionKind)
                self.client.recordSymbolLocation(symbolId, sourceRange)

            if referenceKind is not None:
                referenceId = self.client.recordReference(
                    self.contextStack[-1].id, symbolId, referenceKind
                )
                self.client.recordReferenceLocation(referenceId, sourceRange)
        else:
            localSymbolId = self.client.recordLocalSymbol(
                self.getLocalSymbolName(definition)
            )
            self.client.recordLocalSymbolLocation(localSymbolId, sourceRange)
        return True

    def getLocalSymbolName(self, definition):
        definitionNameNode = definition._name.tree_name

        definitionModulePath = definition.module_path
        if definitionModulePath is None:
            if self.sourceFilePath == _virtualFilePath:
                definitionModulePath = self.sourceFilePath

        contextName = ""
        if definitionModulePath is not None:
            parentFuncdef = getParentWithType(definitionNameNode, "funcdef")
            if parentFuncdef is not None:
                parentFuncdefNameNode = getFirstDirectChildWithType(
                    parentFuncdef, "name"
                )
                if parentFuncdefNameNode is not None:
                    parentFuncdefNameHierarchy = self.getNameHierarchyOfNode(
                        parentFuncdefNameNode, definitionModulePath
                    )
                    if parentFuncdefNameHierarchy is not None:
                        contextName = parentFuncdefNameHierarchy.getDisplayString()

        if len(contextName) == 0:
            contextName = str(self.contextStack[-1].name)

        return contextName + "<" + definitionNameNode.value + ">"

    def getNameHierarchyFromModuleFilePath(self, filePath):
        if filePath is None:
            return None

        if filePath == _virtualFilePath:
            return NameHierarchy(
                NameElement(os.path.splitext(_virtualFilePath)[0]), "."
            )

        filePath = os.path.abspath(filePath)
        filePath = os.path.splitext(filePath)[0]

        sysPath = []

        jediPath = os.path.dirname(jedi.__file__)
        major = self.environment.version_info.major
        minor = self.environment.version_info.minor
        if major == 2:
            sysPath.append(os.path.abspath(jediPath + "/third_party/typeshed/stdlib/2"))
        if major == 2 or major == 3:
            sysPath.append(
                os.path.abspath(jediPath + "/third_party/typeshed/stdlib/2and3")
            )
        if major == 3:
            sysPath.append(os.path.abspath(jediPath + "/third_party/typeshed/stdlib/3"))
            if minor == 5:
                sysPath.append(
                    os.path.abspath(jediPath + "/third_party/typeshed/stdlib/3.5")
                )
            if minor == 6:
                sysPath.append(
                    os.path.abspath(jediPath + "/third_party/typeshed/stdlib/3.6")
                )
            if minor == 7:
                sysPath.append(
                    os.path.abspath(jediPath + "/third_party/typeshed/stdlib/3.7")
                )

        sysPath.extend(self.sysPath)

        for p in sysPath:
            if filePath.startswith(p):
                rest = filePath[len(p) :]
                if rest.startswith(os.path.sep):
                    # Remove a slash in cases it's still there.
                    rest = rest[1:]
                if rest:
                    split = rest.split(os.path.sep)
                    for string in split:
                        if not string:
                            return None

                    if split[-1] == "__init__":
                        split = split[:-1]
                    if split[-1] == "__builtin__":
                        split = split[:-1]
                        split.insert(0, "builtins")

                    nameHierarchy = None
                    for namePart in split:
                        if nameHierarchy is None:
                            nameHierarchy = NameHierarchy(NameElement(namePart), ".")
                        else:
                            nameHierarchy.nameElements.append(NameElement(namePart))
                    return nameHierarchy

        return None

    def getNameHierarchyFromModulePathOfDefinition(self, definition):
        nameHierarchy = self.getNameHierarchyFromModuleFilePath(definition.module_path)
        if nameHierarchy is not None:
            if nameHierarchy.nameElements[-1].name != definition.name:
                nameHierarchy.nameElements.append(NameElement(definition.name))
        return nameHierarchy, definition.module_path

    def getNameHierarchyFromFullNameOfDefinition(self, definition):
        nameHierarchy = None
        for namePart in definition.full_name.split("."):
            if nameHierarchy is None:
                nameHierarchy = NameHierarchy(NameElement(namePart), ".")
            else:
                nameHierarchy.nameElements.append(NameElement(namePart))
        return nameHierarchy

    def getNameHierarchyOfClassOrFunctionDefinition(self, definition):
        if definition is None:
            return None

        if definition.line is None and definition.column is None:
            if definition.module_name in ["builtins", "__builtin__"]:
                nameHierarchy = NameHierarchy(NameElement("builtins"), ".")
                if definition.full_name is not None:
                    for namePart in definition.full_name.split("."):
                        nameHierarchy.nameElements.append(NameElement(namePart))
                else:
                    for namePart in definition.name.split("."):
                        nameHierarchy.nameElements.append(NameElement(namePart))
                return nameHierarchy
            else:
                return self.getNameHierarchyFromFullNameOfDefinition(definition)

        else:
            if definition._name is None or definition._name.tree_name is None:
                return None

            definitionModulePath = definition.module_path
            if definitionModulePath is None:
                if self.sourceFilePath == _virtualFilePath:
                    definitionModulePath = self.sourceFilePath
                else:
                    return None

            return (
                self.getNameHierarchyOfNode(
                    definition._name.tree_name, definitionModulePath
                ),
                definitionModulePath,
                definition._name.tree_name,
            )

    def getDefinitionsOfNode(self, node, nodeSourceFilePath):
        try:
            (startLine, startColumn) = node.start_pos
            script = self.createScript(nodeSourceFilePath)
            return script.goto(line=startLine, column=startColumn, follow_imports=True)

        except Exception:
            return []

    def getNameHierarchyOfNode(self, node, nodeSourceFilePath):
        if node is None:
            return None

        if node.type == "name":
            nameNode = node
        else:
            nameNode = getFirstDirectChildWithType(node, "name")

        if nameNode is None:
            return None

        # we derive the name for the canonical node (e.g. the node's definition)
        for definition in self.getDefinitionsOfNode(nameNode, nodeSourceFilePath):
            if definition is None:
                continue

            definitionModulePath = definition.module_path
            if definitionModulePath is None:
                if self.sourceFilePath == _virtualFilePath:
                    definitionModulePath = self.sourceFilePath
                else:
                    continue

            definitionNameNode = definition._name.tree_name
            if definitionNameNode is None:
                continue

            parentNode = getParentWithTypeInList(
                definitionNameNode.parent, ["classdef", "funcdef"]
            )
            potentialSelfNode = getNamedParentNode(definitionNameNode)
            # if the node is defines as a non-static member variable, we remove the "function_name.self" from the
            # name hierarchy (e.g. "Foo.__init__.self.bar" gets shortened to "Foo.bar")
            if potentialSelfNode is not None:
                potentialSelfNameNode = getFirstDirectChildWithType(
                    potentialSelfNode, "name"
                )
                if potentialSelfNameNode is not None:
                    for potentialSelfDefinition in self.getDefinitionsOfNode(
                        potentialSelfNameNode, definitionModulePath
                    ):
                        if (
                            potentialSelfDefinition is None
                            or potentialSelfDefinition.type != "param"
                        ):
                            continue

                        potentialSelfDefinitionNameNode = (
                            potentialSelfDefinition._name.tree_name
                        )

                        potentialFuncdefNode = getNamedParentNode(
                            potentialSelfDefinitionNameNode
                        )
                        if (
                            potentialFuncdefNode is None
                            or potentialFuncdefNode.type != "funcdef"
                        ):
                            continue

                        potentialClassdefNode = getNamedParentNode(potentialFuncdefNode)
                        if (
                            potentialClassdefNode is None
                            or potentialClassdefNode.type != "classdef"
                        ):
                            continue

                        preceedingNode = (
                            potentialSelfDefinitionNameNode.parent.get_previous_sibling()
                        )
                        if (
                            preceedingNode is not None
                            and preceedingNode.type != "param"
                        ):
                            # 'node' is the first parameter of a member function (aka. 'self')
                            parentNode = potentialClassdefNode

            nameElement = NameElement(definitionNameNode.value)

            if parentNode is not None:
                parentNodeNameHierarchy = self.getNameHierarchyOfNode(
                    parentNode, definitionModulePath
                )
                if parentNodeNameHierarchy is None:
                    return None
                parentNodeNameHierarchy.nameElements.append(nameElement)
                return parentNodeNameHierarchy

            nameHierarchy = self.getNameHierarchyFromModuleFilePath(nodeSourceFilePath)
            if nameHierarchy is None:
                return None
            nameHierarchy.nameElements.append(nameElement)
            return nameHierarchy

        return None

    def createScript(self, sourceFilePath):
        if (
            sourceFilePath == _virtualFilePath
        ):  # we are indexing a provided code snippet
            return SourcetrailScript(
                source=self.sourceFileContent,
                environment=self.environment,
                sys_path=self.sysPath,
            )
        else:  # we are indexing a real file
            return SourcetrailScript(
                source=None,
                path=sourceFilePath,
                environment=self.environment,
                sys_path=self.sysPath,
            )


class VerboseAstVisitor(AstVisitor):

    def __init__(
        self, client, evaluator, sourceFilePath, sourceFileContent=None, sysPath=None
    ):
        AstVisitor.__init__(
            self, client, evaluator, sourceFilePath, sourceFileContent, sysPath
        )
        self.indentationLevel = 0
        self.indentationToken = "| "

    def traverseNode(self, node):
        currentString = ""
        for i in range(0, self.indentationLevel):
            currentString += self.indentationToken

        currentString += node.type

        if hasattr(node, "value"):
            currentString += " (" + repr(node.value) + ")"

        currentString += " " + getSourceRangeOfNode(node).toString()

        print("AST: " + currentString)

        self.indentationLevel += 1
        AstVisitor.traverseNode(self, node)
        self.indentationLevel -= 1
