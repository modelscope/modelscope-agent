from modelscope_agent.register import (action_parser_register,
                                       prompt_generator_register)

from .action_parser_factory import get_action_parser
from .prompt import get_prompt_generator

# PROMPT_MODULES = ["prompt"]
#
# ALL_MODULES = [("modelscope_agent", PROMPT_MODULES)]
#
#
# def _handle_errors(errors):
#     """Log out and possibly reraise errors during import."""
#     if not errors:
#         return
#     for name, err in errors:
#         logging.warning("Module {} import failed: {}".format(name, err))
#
#
# def import_all_modules_for_register(custom_module_paths=None):
#     """Import all modules for register."""
#     modules = []
#     for base_dir, modules in ALL_MODULES:
#         for name in modules:
#             full_name = base_dir + "." + name
#             modules.append(full_name)
#     if isinstance(custom_module_paths, list):
#         modules += custom_module_paths
#     errors = []
#     for module in modules:
#         try:
#             importlib.import_module(module)
#         except ImportError as error:
#             errors.append((module, error))
#     _handle_errors(errors)
#
#
# import_all_modules_for_register(custom_module_paths='apps/agentfabric')
