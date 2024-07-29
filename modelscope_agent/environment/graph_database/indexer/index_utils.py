import jedi
import json
from jedi.api import classes, helpers
from jedi.inference.gradual.conversion import convert_names

_virtualFilePath = 'virtual_file.py'


class SourcetrailScript(jedi.Script):

    def __init__(
        self,
        source=None,
        line=None,
        column=None,
        path=None,
        encoding='utf-8',
        sys_path=None,
        environment=None,
        _project=None,
    ):
        jedi.Script.__init__(self, source, line, column, path, encoding,
                             sys_path, environment, _project)

    def _goto(
        self,
        line,
        column,
        follow_imports=False,
        follow_builtin_imports=False,
        only_stubs=False,
        prefer_stubs=False,
        follow_override=False,
    ):
        if follow_override:
            return super()._goto(
                line,
                column,
                follow_imports=follow_imports,
                follow_builtin_imports=follow_builtin_imports,
                only_stubs=only_stubs,
                prefer_stubs=prefer_stubs,
            )
        tree_name = self._module_node.get_name_of_position((line, column))
        if tree_name is None:
            # Without a name we really just want to jump to the result e.g.
            # executed by `foo()`, if we the cursor is after `)`.
            return self.infer(
                line, column, only_stubs=only_stubs, prefer_stubs=prefer_stubs)
        name = self._get_module_context().create_name(tree_name)

        names = list(name.goto())

        if follow_imports:
            names = helpers.filter_follow_imports(names,
                                                  follow_builtin_imports)
        names = convert_names(
            names,
            only_stubs=only_stubs,
            prefer_stubs=prefer_stubs,
        )

        defs = [classes.Name(self._inference_state, d) for d in set(names)]
        return list(set(helpers.sorted_definitions(defs)))


class ContextInfo:

    def __init__(self, id, name, node):
        self.id = id
        self.name = name
        self.node = node


class SourceRange:

    def __init__(self, startLine, startColumn, endLine, endColumn):
        self.startLine = startLine
        self.startColumn = startColumn
        self.endLine = endLine
        self.endColumn = endColumn

    def toString(self):
        return ('[' + str(self.startLine) + ':' + str(self.startColumn) + '|'
                + str(self.endLine) + ':' + str(self.endColumn) + ']')


class NameHierarchy:

    unsolvedSymbolName = 'unsolved symbol'
    base_element_list = []

    def __init__(self, nameElement, delimiter, copy=False):

        self.nameElements = []
        if nameElement is not None:
            self.nameElements.append(nameElement)
        self.delimiter = delimiter

    def copy(self):
        ret = NameHierarchy(None, self.delimiter, copy=True)
        for nameElement in self.nameElements:
            ret.nameElements.append(
                NameElement(nameElement.name, nameElement.prefix,
                            nameElement.postfix))
        return ret

    def serialize(self):
        return json.dumps(self, cls=NameHierarchyEncoder)

    def getDisplayString(self):
        displayString = ''
        isFirst = True
        for nameElement in self.nameElements:
            if not isFirst:
                displayString += self.delimiter
            isFirst = False
            if len(nameElement.prefix) > 0:
                displayString += nameElement.prefix + ' '
            displayString += nameElement.name
            if len(nameElement.postfix) > 0:
                displayString += nameElement.postfix
        return displayString

    def getParentDisplayString(self):
        displayString = ''
        isFirst = True
        for nameElement in self.nameElements[:-1]:
            if not isFirst:
                displayString += self.delimiter
            isFirst = False
            if len(nameElement.prefix) > 0:
                displayString += nameElement.prefix + ' '
            displayString += nameElement.name
            if len(nameElement.postfix) > 0:
                displayString += nameElement.postfix
        return displayString


class NameElement:

    def __init__(self, name, prefix='', postfix=''):
        self.name = name
        self.prefix = prefix
        self.postfix = postfix


class NameHierarchyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, NameHierarchy):
            return {
                'name_delimiter':
                obj.delimiter,
                'name_elements':
                [nameElement.__dict__ for nameElement in obj.nameElements],
            }
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def getNameHierarchyForUnsolvedSymbol():
    return NameHierarchy(NameElement(NameHierarchy.unsolvedSymbolName), '')


def isQualifierNode(node):
    nextNode = getNext(node)
    if nextNode is not None and nextNode.type == 'trailer':
        nextNode = getNext(nextNode)
    if nextNode is not None and nextNode.type == 'operator' and nextNode.value == '.':
        return True
    return False


def isCallNode(node):
    nextNode = getNext(node)
    if nextNode is not None and nextNode.type == 'trailer':
        if (len(nextNode.children) >= 2 and nextNode.children[0].value == '('
                and nextNode.children[-1].value == ')'):
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
