import logging
from typing import List, Dict, Union
from tree_sitter import Node, Language, Parser
import tree_sitter_python as tspython
from config import PYTHON_BUILTINS

class CodeParser:
    def __init__(self):
        self.language_name = "python"
        self.parser = None
        self.language = None
        logging.info(f"Attempting to load parser for '{self.language_name}' using official method...")
        self._load_parser()

    def _load_parser(self):
        try:
            logging.info("  Loading Language object from tree_sitter_python...")
            self.language = Language(tspython.language())
            logging.info(f"  Successfully created Language object: {type(self.language)}")
            logging.info("  Initializing Parser with the Language object...")
            self.parser = Parser(self.language)
            logging.info(f"  Successfully initialized Parser object: {type(self.parser)}")

            test_code = "print('hello')"
            logging.info(f"  Attempting to parse test string: '{test_code}'")
            tree = self.parser.parse(bytes(test_code, "utf8"))
            if tree and tree.root_node:
                logging.info(f"  Test parse successful. Root node: {tree.root_node.type}")
                logging.info(f"Successfully loaded and tested parser for {self.language_name}")
            else:
                logging.warning("  Test parse failed or returned empty tree.")
        except ImportError as e:
            logging.error(f"Failed to import 'tree_sitter_python' or its components.")
            logging.error(f"  Ensure 'tree-sitter-python' is installed (`pip install tree-sitter-python`).")
            logging.error(f"  ImportError: {e}")
            self.parser = None
        except Exception as e:
            logging.error(f"Failed to load or test parser for '{self.language_name}'.")
            logging.error(f"  Exception Type: {type(e)}")
            logging.error(f"  Error Details: {e}")
            self.parser = None

    def is_parser_loaded(self) -> bool:
        return self.parser is not None

    def parse_code(self, code: str) -> Union[None, Node]:
        if not self.is_parser_loaded(): return None
        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            return tree.root_node if tree and tree.root_node else None
        except Exception as e:
            logging.error(f"Error during parsing: {e}")
            return None

    def _traverse_tree(self, node: Node, code: str, file_path: str, entities: List[Dict]):
        if node is None: return
        node_type_map = self._get_python_node_types_of_interest_for_entities()
        entity_type = node_type_map.get(node.type)
        if entity_type:
            entity_name = self._get_entity_name(node, code, entity_type)
            start_line = node.start_point[0] + 1; end_line = node.end_point[0] + 1
            snippet = code[node.start_byte:node.end_byte]
            entity_data = {"entity_type": entity_type, "name": entity_name, "source_file": file_path,
                           "start_line": start_line, "end_line": end_line, "start_byte": node.start_byte,
                           "end_byte": node.end_byte, "snippet": snippet}
            parent_info = self._find_parent_context(node, code)
            if parent_info: entity_data["parent_type"], entity_data["parent_name"] = parent_info["type"], parent_info["name"]
            if entity_type == "ImportFrom": entity_data.update(self._extract_import_from_details(node, code))
            elif entity_type == "Import": entity_data['module_name'] = entity_name
            entities.append(entity_data)
        try:
            for child in node.children:
                self._traverse_tree(child, code, file_path, entities)
        except Exception as e:
            logging.warning(f"Error traversing children of node type {node.type} at {file_path}:{node.start_point}: {e}")

    def extract_entities_from_file(self, code: str, file_path: str) -> List[Dict]:
        root_node = self.parse_code(code); entities = []
        if root_node: self._traverse_tree(root_node, code, file_path, entities)
        return entities

    def _get_python_node_types_of_interest_for_entities(self) -> Dict[str, str]:
        return {
            'import_statement': 'Import',
            'import_from_statement': 'ImportFrom',
            'class_definition': 'Class',
            'function_definition': 'Function',
            'decorated_definition': 'DecoratedDefinition',
            'call': 'Call',
        }

    def _get_qualified_call_name(self, node: Node, code: str) -> str:
        if node is None: return ""
        if node.type == 'identifier':
            return code[node.start_byte:node.end_byte]
        elif node.type == 'attribute':
            obj_node = node.child_by_field_name('object')
            attr_node = node.child_by_field_name('attribute')
            if obj_node and attr_node:
                base = self._get_qualified_call_name(obj_node, code)
                attr = code[attr_node.start_byte:attr_node.end_byte]
                return f"{base}.{attr}" if base else attr
            else:
                logging.debug(f"Unexpected attribute structure at {node.start_point}. Falling back.")
                return code[node.start_byte:node.end_byte]
        elif node.type == 'call':
            func_node = node.child_by_field_name('function')
            if func_node:
                return self._get_qualified_call_name(func_node, code)
            else:
                logging.debug(f"Unexpected call structure at {node.start_point}. Falling back.")
                return code[node.start_byte:node.end_byte]
        else:
            logging.debug(f"Unhandled node type {node.type} in qualified call name extraction at {node.start_point}. Falling back.")
            return code[node.start_byte:node.end_byte]

    def _get_entity_name(self, node: Node, code: str, node_type: str) -> str:
        name = f"Unnamed_{node_type}_{node.start_byte}"
        try:
            if node_type in ["Function", "Class"]:
                name_node = node.child_by_field_name('name')
                if name_node and name_node.type == 'identifier':
                    name = code[name_node.start_byte:name_node.end_byte]
            elif node_type == "DecoratedDefinition":
                def_node = next((c for c in node.children if c.type in ['function_definition', 'class_definition']), None)
                if def_node:
                    inner_type = "Function" if def_node.type == "function_definition" else "Class"
                    name = self._get_entity_name(def_node, code, inner_type)
            elif node_type == "Call":
                func_node = node.child_by_field_name('function')
                if func_node:
                    name = self._get_qualified_call_name(func_node, code)
                else:
                    name = code[node.start_byte:node.end_byte].split('(')[0].strip()
            elif node_type == "Import":
                name_node = next((c for c in node.children if c.type in ['dotted_name', 'aliased_import']), None)
                if name_node:
                    if name_node.type == 'dotted_name':
                        name = code[name_node.start_byte:name_node.end_byte]
                    elif name_node.type == 'aliased_import':
                        original_name_node = name_node.child_by_field_name('name')
                        if original_name_node:
                            name = code[original_name_node.start_byte:original_name_node.end_byte]
                        else:
                            alias_node = name_node.child_by_field_name('alias')
                            if alias_node: name = code[alias_node.start_byte:alias_node.end_byte]
                else:
                    name = code[node.start_byte:node.end_byte].replace("import ", "").strip().split(' ')[0]
            elif node_type == "ImportFrom":
                module_node = node.child_by_field_name('module_name')
                if module_node:
                    name = code[module_node.start_byte:module_node.end_byte]
        except Exception as e:
            logging.warning(f"Error extracting name for {node_type} at {node.start_point}: {e}", exc_info=True)

        safe_name = "".join(c if c.isalnum() or c in ['_', '.', '-'] else '_' for c in name)
        safe_name = '.'.join(filter(None, safe_name.split('.')))
        return safe_name if safe_name else f"Unnamed_{node_type}_{node.start_byte}"

    def _find_parent_context(self, node: Node, code: str) -> Union[Dict, None]:
        current = node.parent
        while current:
            p_type = current.type
            ctx_type = None
            name = None
            try:
                if p_type == "function_definition":
                    ctx_type = "Function"
                    name = self._get_entity_name(current, code, ctx_type)
                elif p_type == "class_definition":
                    ctx_type = "Class"
                    name = self._get_entity_name(current, code, ctx_type)
                elif p_type == "decorated_definition":
                    inner_def = next((c for c in current.children if c.type in ["function_definition", "class_definition"]), None)
                    if inner_def:
                        inner_type = "Function" if inner_def.type == "function_definition" else "Class"
                        name = self._get_entity_name(inner_def, code, inner_type)
                        if node.parent == current:
                            return {"type": inner_type, "name": name}
                    current = current.parent
                    continue
                if ctx_type and name:
                    return {"type": ctx_type, "name": name}
            except Exception as e:
                logging.warning(f"Error finding parent context name for node type {p_type}: {e}")
            current = current.parent
        return None

    def _extract_import_from_details(self, node: Node, code: str) -> Dict:
        details = {"imported_items": [], "module_name": "UnknownModule"}
        mod_node = node.child_by_field_name('module_name')
        if mod_node:
            details['module_name'] = code[mod_node.start_byte:mod_node.end_byte]

        names_part = next((c for c in node.children if c.type in ['import_list', 'aliased_import', 'dotted_name', 'wildcard_import']), None)
        if names_part:
            if names_part.type == 'wildcard_import':
                details['imported_items'].append('*')
            else:
                self._find_python_imported_names(names_part, code, details['imported_items'])
        elif node.named_child_count > 1:
            potential_name_node = node.named_children[-1]
            if potential_name_node.type in ['identifier', 'dotted_name']:
                self._find_python_imported_names(potential_name_node, code, details['imported_items'])
        return details

    def _find_python_imported_names(self, node: Node, code: str, items_list: List[str]):
        try:
            if node.type == 'identifier':
                parent_type = node.parent.type if node.parent else None
                if parent_type in ['import_list', 'aliased_import', None]:
                    items_list.append(code[node.start_byte:node.end_byte])
                elif parent_type == 'dotted_name' and node == node.parent.children[-1]:
                    grandparent_type = node.parent.parent.type if node.parent.parent else None
                    if grandparent_type == 'import_list':
                        pass
            elif node.type == 'dotted_name':
                parent_type = node.parent.type if node.parent else None
                if parent_type == 'import_list':
                    items_list.append(code[node.start_byte:node.end_byte])
                elif parent_type == 'aliased_import' and node == node.parent.child_by_field_name('name'):
                    pass
                elif parent_type not in ['attribute', 'call']:
                    items_list.append(code[node.start_byte:node.end_byte])
            elif node.type == 'aliased_import':
                orig_node = node.child_by_field_name('name')
                alias_node = node.child_by_field_name('alias')
                o_name = code[orig_node.start_byte:orig_node.end_byte] if orig_node else '??'
                a_name = code[alias_node.start_byte:alias_node.end_byte] if alias_node else '??'
                items_list.append(f"{o_name} as {a_name}")
            else:
                if node.type != 'aliased_import':
                    for child in node.children:
                        self._find_python_imported_names(child, code, items_list)
        except Exception as e:
            logging.warning(f"Error finding imported names in node type {node.type}: {e}", exc_info=True)

    def _get_python_node_types_of_interest_for_chunking(self) -> Dict[str, str]:
        return {
            'import_statement': 'Import',
            'import_from_statement': 'ImportFrom',
            'class_definition': 'Class',
            'function_definition': 'Function',
            'decorated_definition': 'DecoratedDefinition',
        }

    def _get_python_nodes_for_comments(self) -> Dict[str, str]:
        return {
            'comment': 'Comment',
            'decorator': 'Decorator',
        }

    def _traverse_for_lines(self, node: Node, types_of_interest: Dict[str, str], lines_found: List[int]):
        if node is None:
            return
        if node.type in types_of_interest:
            lines_found.append(node.start_point[0])
        for child in node.children:
            self._traverse_for_lines(child, types_of_interest, lines_found)

    def extract_lines_for_points_of_interest(self, root_node: Node) -> List[int]:
        if not root_node:
            return []
        node_types = self._get_python_node_types_of_interest_for_chunking()
        lines_found = []
        self._traverse_for_lines(root_node, node_types, lines_found)
        return sorted(list(set(lines_found)))

    def extract_lines_for_comments(self, root_node: Node) -> List[int]:
        if not root_node:
            return []
        node_types = self._get_python_nodes_for_comments()
        lines_found = []
        self._traverse_for_lines(root_node, node_types, lines_found)
        return sorted(list(set(lines_found)))