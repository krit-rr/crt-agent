from crt_agent.agents.ablation import AblationAgent
from crt_agent.agents.base import Agent, AgentState
from crt_agent.agents.dual_process import DualProcessAgent
from crt_agent.agents.symbolic_agent import SymbolicAgent
from crt_agent.agents.tool_agent import ToolAgent

AGENTS: dict[str, type[Agent]] = {
    cls.name: cls for cls in (SymbolicAgent, ToolAgent, DualProcessAgent, AblationAgent)
}

__all__ = [
    "AGENTS",
    "AblationAgent",
    "Agent",
    "AgentState",
    "DualProcessAgent",
    "SymbolicAgent",
    "ToolAgent",
]
