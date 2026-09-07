"""Permission module — dual-layer permission control for tool calls.

Outer layer (PermissionEnforcer): user-intent based, configurable, overridable.
Inner layer (SafetyGuard): safety baseline, non-bypassable.
"""

from .ask_resolver import resolve_ask
from .ask_options import AskOption, build_ask_options, parse_ask_choice
from .approval import (ApprovalConflictError, ApprovalRequest, ApprovalStore,
                       FileApprovalStore, MemoryApprovalStore)
from .config import PermissionConfig, SafetyConfig
from .enforcer import PermissionDecision, PermissionEnforcer
from .handler import (AutoPermissionHandler, CLIPermissionHandler,
                      PermissionAction, PermissionHandler, PermissionResponse,
                      WebPermissionHandler)
from .memory import PermissionMemory
from .provider import (AgentDecisionProvider, LlmDecisionProvider,
                       PermissionDecisionProvider, ProviderDecision,
                       request_provider_decision)
from .safety import SafetyGuard

__all__ = [
    'resolve_ask',
    'AskOption',
    'build_ask_options',
    'parse_ask_choice',
    'ApprovalConflictError',
    'ApprovalRequest',
    'ApprovalStore',
    'FileApprovalStore',
    'MemoryApprovalStore',
    'PermissionConfig',
    'SafetyConfig',
    'PermissionDecision',
    'PermissionEnforcer',
    'AutoPermissionHandler',
    'CLIPermissionHandler',
    'PermissionAction',
    'PermissionHandler',
    'PermissionResponse',
    'WebPermissionHandler',
    'PermissionMemory',
    'AgentDecisionProvider',
    'LlmDecisionProvider',
    'PermissionDecisionProvider',
    'ProviderDecision',
    'request_provider_decision',
    'SafetyGuard',
]
