import codecs

import sourcetraildb as srctrl
from my_graph_db import GraphDatabaseHandler


class SymbolRegistry:

    def __init__(self):
        self.symbol_dict = {}
        self.next_id = 1

    def record_symbol(self, symbol_name):
        if symbol_name not in self.symbol_dict:
            self.symbol_dict[symbol_name] = self.next_id
            self.next_id += 1
        return self.symbol_dict[symbol_name]


class SymbolReferenceRegistry:

    def __init__(self):
        self.references = {}
        self.next_ref_id = 1

    def record_reference(self, context_symbol_id, referenced_symbol_id,
                         reference_kind):
        # 生成一个关系键值，用于在字典中查找
        reference_key = (context_symbol_id, referenced_symbol_id,
                         reference_kind)

        # 检查是否存在重复的关系
        if reference_key in self.references:
            return self.references[reference_key]

        # 如果不存在重复关系，则创建新的关系ID
        reference_id = self.next_ref_id
        self.references[reference_key] = reference_id
        self.next_ref_id += 1
        return reference_id


class AstVisitorClient:

    def __init__(self, graphDB: GraphDatabaseHandler, task_root_path=''):
        self.indexedFileId = 0
        self.symbol = SymbolRegistry()
        self.symbol_rela = SymbolReferenceRegistry()
        self.task_root_path = task_root_path
        self.graphDB = graphDB
        self.this_module = ''
        self.this_file_path = ''
        self.this_script = None
        self.this_source_code_lines = []
        # self.graphDB.clear_database()
        self.symbol_data = {}
        self.symbol_data['builtins'] = {
            'name': 'builtins',
            'kind': 'MODULE',
            'parent_name': '',
        }
        self.symbolId_to_Name = {}
        self.indexedFileId_to_path = {}
        self.referenceId_to_data = {}
        self.referenceId_to_data['Unsolved'] = []

    def process_file_path(self, file_path):
        if self.task_root_path:
            if file_path.startswith(self.task_root_path):
                file_path = file_path[len(self.task_root_path):]
        return file_path

    def extract_signature(self, code):
        pass

    def extract_code_between_lines(self,
                                   start_line,
                                   end_line,
                                   is_indent=True,
                                   is_code=True):
        if is_code:
            return '<CODE>{{"S":{0},"E":{1},"F":"{2}"}}</CODE>'.format(
                start_line, end_line, self.this_file_path)
        if start_line < 1:
            start_line = 1
        extracted_lines = self.this_source_code_lines[start_line - 1:end_line]

        # 去除指定数量的缩进
        if is_indent:
            extracted_lines = self.this_source_code_lines[start_line
                                                          - 1:end_line]
            first_line_indent = len(extracted_lines[0]) - len(
                extracted_lines[0].lstrip())

            extracted_lines = [
                line[first_line_indent:]
                if len(line) > first_line_indent else ''
                for line in extracted_lines
            ]

        extracted_code = '\n'.join(extracted_lines)
        return extracted_code

    def extract_code_from_file(self,
                               file_path,
                               start_line,
                               end_line,
                               is_indent=True,
                               is_code=True):
        if is_code:
            return '<CODE>{{"S":{0},"E":{1},"F":"{2}"}}</CODE>'.format(
                start_line, end_line, file_path)
        if start_line < 1:
            start_line = 1
        try:
            with codecs.open(file_path, 'r', encoding='utf-8') as input:
                sourceCode = input.read()
            source_code_lines = sourceCode.split('\n')
            extracted_lines = source_code_lines[start_line - 1:end_line]
        except Exception:
            return ''
        # 去除指定数量的缩进
        if is_indent:
            first_line_indent = len(extracted_lines[0]) - len(
                extracted_lines[0].lstrip())

            extracted_lines = [
                line[first_line_indent:]
                if len(line) > first_line_indent else ''
                for line in extracted_lines
            ]

        extracted_code = '\n'.join(extracted_lines)
        return extracted_code

    def get_module_name(self, symbol):
        if symbol not in self.symbol_data.keys():
            return ''
        if self.symbol_data[symbol]['kind'] == 'MODULE':
            return symbol
        return self.get_module_name(self.symbol_data[symbol]['parent_name'])

    def get_parent_class(self, symbol):
        parent_name = self.symbol_data[symbol]['parent_name']
        if parent_name in self.symbol_data.keys():
            parent_type = self.symbol_data[parent_name]['kind']
            if parent_type == 'CLASS':
                return parent_name
        return ''

    def get_import_scope_location(self, tree_node):
        try:
            start_pos = tree_node.parent.start_pos[0]
            end_pos = tree_node.parent.end_pos[0]
            return start_pos, end_pos
        except Exception:
            return -1, -1

    def recordSymbol(self,
                     nameHierarchy,
                     node_path='',
                     tree_node=None,
                     global_node=None):

        if nameHierarchy is not None:
            name = nameHierarchy.getDisplayString()
            parent_name = nameHierarchy.getParentDisplayString()
            symbolId = self.symbol.record_symbol(name)
            self.symbolId_to_Name[symbolId] = name
            if name not in self.symbol_data.keys():
                self.symbol_data[name] = {
                    'name': name,
                    'path': node_path,
                    'kind': '',
                    'parent_name': parent_name,
                }
            if global_node:
                start_line, end_line = self.get_import_scope_location(
                    global_node)
                code = self.extract_code_from_file(
                    node_path, start_line - 2, end_line + 2, is_indent=True)
                self.symbol_data[name]['code'] = code
            if tree_node and node_path:
                start_line, end_line = self.get_import_scope_location(
                    tree_node)
                code = self.extract_code_from_file(
                    node_path, start_line, end_line, is_indent=True)
                self.symbol_data[name]['code'] = code
            # self.symbol_data[name] = {
            # 	"name": name,
            # 	"path": node_path,
            # 	"kind": '',
            # 	"code": code,
            # 	"parent_name": parent_name,
            # }
            return symbolId
        return 0

    def recordSymbolDefinitionKind(self, symbolId, symbolDefinitionKind):
        # definition 的类型
        pass

    def recordSymbolKind(self, symbolId, symbolKind):
        full_name = self.symbolId_to_Name[symbolId]
        kind = symbolKindToString(symbolKind)
        self.symbol_data[full_name]['kind'] = kind
        # create node
        if kind == 'MODULE':
            if full_name == self.this_module:
                self.graphDB.add_node(
                    label='MODULE',
                    full_name=full_name,
                    parms={
                        'name': full_name,
                        'file_path':
                        self.process_file_path(self.this_file_path),
                    },
                )
            else:
                self.graphDB.add_node(
                    label='MODULE',
                    full_name=full_name,
                    parms={
                        'name':
                        full_name,
                        'file_path':
                        self.process_file_path(
                            self.symbol_data[full_name]['path']),
                    },
                )
        elif kind in [
                'CLASS', 'FUNCTION', 'METHOD', 'GLOBAL_VARIABLE', 'FIELD'
        ]:

            data = {
                'name':
                full_name.split('.')[-1],
                'file_path':
                self.process_file_path(self.symbol_data[full_name]['path']),
            }
            if self.symbol_data[full_name]['parent_name'] == self.this_module:
                data['file_path'] = self.process_file_path(self.this_file_path)
            if 'code' in self.symbol_data[full_name].keys():
                data['code'] = self.symbol_data[full_name]['code']

            if kind in ['FUNCTION', 'METHOD', 'GLOBAL_VARIABLE', 'FIELD']:
                parent_class = self.get_parent_class(full_name)
                if parent_class:
                    data['class'] = parent_class
                    parent_name = self.symbol_data[parent_class]['parent_name']
                    if parent_name == self.this_module:
                        data['file_path'] = self.process_file_path(
                            self.this_file_path)
                    if kind == 'FUNCTION':
                        kind = 'METHOD'
                        self.symbol_data[full_name]['kind'] = kind
            # 创建节点 ------------------------------------------------------------------
            self.graphDB.add_node(label=kind, full_name=full_name, parms=data)
            # 边的关系 ------------------------------------------------------------------
            if kind in ['CLASS', 'FUNCTION', 'GLOBAL_VARIABLE']:
                module_name = self.get_module_name(full_name)
                self.graphDB.add_edge(
                    start_label='MODULE',
                    start_name=module_name,
                    relationship_type='CONTAINS',
                    end_label=kind,
                    end_name=full_name,
                    params={'association_type': kind},
                )
                self.graphDB.add_edge(
                    start_label='MODULE',
                    start_name=self.this_module,
                    relationship_type='CONTAINS',
                    end_label=kind,
                    end_name=full_name,
                    params={'association_type': kind},
                )
            if kind == 'METHOD':
                parent_class = self.get_parent_class(full_name)
                self.graphDB.add_edge(
                    start_label='CLASS',
                    start_name=parent_class,
                    relationship_type='HAS_METHOD',
                    end_label=kind,
                    end_name=full_name,
                )
            if kind == 'FIELD':
                parent_class = self.get_parent_class(full_name)
                self.graphDB.add_edge(
                    start_label='CLASS',
                    start_name=parent_class,
                    relationship_type='HAS_FIELD',
                    end_label=kind,
                    end_name=full_name,
                )

    def recordSymbolLocation(self, symbolId, sourceRange):
        name = self.symbolId_to_Name[symbolId]
        kind = self.symbol_data[name]['kind']

        if kind in ['CLASS', 'FUNCTION', 'METHOD']:
            code = self.extract_code_between_lines(
                sourceRange.startLine, sourceRange.endLine, is_code=False)
            self.graphDB.add_node(
                kind, full_name=name, parms={'signature': code.strip()})

        if kind in ['GLOBAL_VARIABLE', 'FIELD']:
            code = self.extract_code_between_lines(sourceRange.startLine - 3,
                                                   sourceRange.endLine + 3)
            self.graphDB.add_node(
                kind, full_name=name, parms={'code': code.strip()})

    def recordSymbolScopeLocation(self, symbolId, sourceRange):
        name = self.symbolId_to_Name[symbolId]
        kind = self.symbol_data[name]['kind']

        if kind in ['CLASS', 'FUNCTION', 'METHOD']:
            code = self.extract_code_between_lines(
                sourceRange.startLine, sourceRange.endLine, is_indent=True)
            self.graphDB.add_node(kind, full_name=name, parms={'code': code})
            self.extract_signature(code)

    def recordSymbolSignatureLocation(self, symbolId, sourceRange):
        pass

    def recordReference(self, contextSymbolId, referencedSymbolId,
                        referenceKind):
        referenceKindStr = referenceKindToString(referenceKind)
        referenceName = self.symbolId_to_Name[referencedSymbolId]
        contextName = self.symbolId_to_Name[contextSymbolId]

        if referenceKindStr == 'IMPORT':
            contextKind = self.symbol_data[contextName]['kind']
            referenceNameKind = self.symbol_data[referenceName]['kind']

        if referenceKindStr == 'CALL':
            contextKind = self.symbol_data[contextName]['kind']
            referenceNameKind = self.symbol_data[referenceName]['kind']
            if contextKind != 'MODULE':
                self.graphDB.add_edge(
                    start_label=contextKind,
                    start_name=contextName,
                    relationship_type='CALL',
                    end_label=referenceNameKind,
                    end_name=referenceName,
                )

        if referenceKindStr in ['USAGE']:
            contextKind = self.symbol_data[contextName]['kind']
            referenceNameKind = self.symbol_data[referenceName]['kind']
            if contextKind in ['FUNCTION', 'METHOD'] and referenceNameKind in [
                    'GLOBAL_VARIABLE',
                    'FIELD',
            ]:
                self.graphDB.add_edge(
                    start_label=contextKind,
                    start_name=contextName,
                    relationship_type='USES',
                    end_label=referenceNameKind,
                    end_name=referenceName,
                )
        if referenceKindStr == 'INHERITANCE':
            contextKind = self.symbol_data[contextName]['kind']
            referenceNameKind = self.symbol_data[referenceName]['kind']
            self.graphDB.add_edge(
                start_label=contextKind,
                start_name=contextName,
                relationship_type='INHERITS',
                end_label=referenceNameKind,
                end_name=referenceName,
            )
        referenceId = self.symbol_rela.record_reference(
            contextSymbolId, referencedSymbolId, referenceKind)
        return referenceId

    def recordReferenceLocation(self, referenceId, sourceRange):

        pass

    def recordReferenceIsAmbiguous(self, referenceId):

        pass

    def recordReferenceToUnsolvedSymhol(self, contextSymbolId, referenceKind,
                                        sourceRange):
        pass

    def recordQualifierLocation(self, referencedSymbolId, sourceRange):
        pass

    def recordFile(self, filePath):
        self.indexedFileId = 1
        self.indexedFileId_to_path[self.indexedFileId] = filePath.replace(
            '\\', '/')
        self.this_file_path = self.indexedFileId_to_path[self.indexedFileId]
        return 1

    def recordFileLanguage(self, fileId, languageIdentifier):
        pass

    def recordLocalSymbol(self, name):
        pass

    def recordLocalSymbolLocation(self, localSymbolId, sourceRange):
        pass

    def recordAtomicSourceRange(self, sourceRange):
        pass

    def recordError(self, message, fatal, sourceRange):
        pass


def symbolDefinitionKindToString(symbolDefinitionKind):
    if symbolDefinitionKind == srctrl.SYMBOL_ANNOTATION:
        return 'EXPLICIT'
    if symbolDefinitionKind == srctrl.DEFINITION_IMPLICIT:
        return 'IMPLICIT'
    return ''


def symbolKindToString(symbolKind):
    if symbolKind == srctrl.SYMBOL_TYPE:
        return 'TYPE'
    if symbolKind == srctrl.SYMBOL_BUILTIN_TYPE:
        return 'BUILTIN_TYPE'
    if symbolKind == srctrl.SYMBOL_MODULE:
        return 'MODULE'
    if symbolKind == srctrl.SYMBOL_NAMESPACE:
        return 'NAMESPACE'
    if symbolKind == srctrl.SYMBOL_PACKAGE:
        return 'PACKAGE'
    if symbolKind == srctrl.SYMBOL_STRUCT:
        return 'STRUCT'
    if symbolKind == srctrl.SYMBOL_CLASS:
        return 'CLASS'
    if symbolKind == srctrl.SYMBOL_INTERFACE:
        return 'INTERFACE'
    if symbolKind == srctrl.SYMBOL_ANNOTATION:
        return 'ANNOTATION'
    if symbolKind == srctrl.SYMBOL_GLOBAL_VARIABLE:
        return 'GLOBAL_VARIABLE'
    if symbolKind == srctrl.SYMBOL_FIELD:
        return 'FIELD'
    if symbolKind == srctrl.SYMBOL_FUNCTION:
        return 'FUNCTION'
    if symbolKind == srctrl.SYMBOL_METHOD:
        return 'METHOD'
    if symbolKind == srctrl.SYMBOL_ENUM:
        return 'ENUM'
    if symbolKind == srctrl.SYMBOL_ENUM_CONSTANT:
        return 'ENUM_CONSTANT'
    if symbolKind == srctrl.SYMBOL_TYPEDEF:
        return 'TYPEDEF'
    if symbolKind == srctrl.SYMBOL_TYPE_PARAMETER:
        return 'TYPE_PARAMETER'
    if symbolKind == srctrl.SYMBOL_FILE:
        return 'FILE'
    if symbolKind == srctrl.SYMBOL_MACRO:
        return 'MACRO'
    if symbolKind == srctrl.SYMBOL_UNION:
        return 'UNION'
    return ''


def referenceKindToString(referenceKind):
    if referenceKind == srctrl.REFERENCE_TYPE_USAGE:
        return 'TYPE_USAGE'
    if referenceKind == srctrl.REFERENCE_USAGE:
        return 'USAGE'
    if referenceKind == srctrl.REFERENCE_CALL:
        return 'CALL'
    if referenceKind == srctrl.REFERENCE_INHERITANCE:
        return 'INHERITANCE'
    if referenceKind == srctrl.REFERENCE_OVERRIDE:
        return 'OVERRIDE'
    if referenceKind == srctrl.REFERENCE_TYPE_ARGUMENT:
        return 'TYPE_ARGUMENT'
    if referenceKind == srctrl.REFERENCE_TEMPLATE_SPECIALIZATION:
        return 'TEMPLATE_SPECIALIZATION'
    if referenceKind == srctrl.REFERENCE_INCLUDE:
        return 'INCLUDE'
    if referenceKind == srctrl.REFERENCE_IMPORT:
        return 'IMPORT'
    if referenceKind == srctrl.REFERENCE_MACRO_USAGE:
        return 'MACRO_USAGE'
    if referenceKind == srctrl.REFERENCE_ANNOTATION_USAGE:
        return 'ANNOTATION_USAGE'
    return ''
