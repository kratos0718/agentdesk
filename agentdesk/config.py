"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    # LLM. Any OpenAI-compatible endpoint works: OpenAI, Groq, Ollama, vLLM.
    # With no API key set the system runs in offline mode (rule-based planner).
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"))
    # USD per 1M tokens, used for the cost estimate in metrics and evals.
    llm_input_cost: float = field(default_factory=lambda: float(os.getenv("LLM_INPUT_COST", "0.59")))
    llm_output_cost: float = field(default_factory=lambda: float(os.getenv("LLM_OUTPUT_COST", "0.79")))

    db_path: Path = field(default_factory=lambda: Path(os.getenv("AGENTDESK_DB", ROOT / "data" / "agentdesk.db")))
    knowledge_dir: Path = ROOT / "data" / "knowledge"

    # Business rules
    refund_auto_limit: float = 5000.0   # refunds above this need a supervisor
    max_steps: int = 6                  # hard cap on tool calls per request

    # API keys -> role. In production these would come from a secrets store.
    api_keys: dict[str, str] = field(default_factory=lambda: {
        os.getenv("AGENTDESK_AGENT_KEY", "agent-dev-key"): "agent",
        os.getenv("AGENTDESK_SUPERVISOR_KEY", "supervisor-dev-key"): "supervisor",
    })

    @property
    def online(self) -> bool:
        return bool(self.llm_api_key)


settings = Settings()
