"""
VLM ajan fabrikası — backend seçimi.

- smolvlm  : yerel geliştirme (RTX 4050 varsayılan)
- internvl2: hedef / Colab T4
- mock     : test / CI
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from src.vlm.internvl_agent import (
    InternVLAgent,
    make_mock_generator,
)
from src.vlm.tools import ToolRegistry

VLMBackend = Literal["smolvlm", "internvl2", "mock"]

DEFAULT_SMOLVLM_ID = "HuggingFaceTB/SmolVLM-Instruct"
DEFAULT_INTERNVL_ID = "OpenGVLab/InternVL2-8B"


def create_vlm_agent(
    backend: VLMBackend = "smolvlm",
    tools: Optional[ToolRegistry] = None,
    model_id: Optional[str] = None,
    device: str = "cuda",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    auto_load: bool = True,
    use_vllm: bool = False,
    **kwargs: Any,
) -> InternVLAgent:
    """
    Backend'e göre yapılandırılmış InternVLAgent (ortak API) döndür.

    InternVLAgent sınıfı hem InternVL hem SmolVLM yolunu `backend` alanı ile yönetir.
    """
    tools = tools or ToolRegistry()
    backend = backend.lower().strip()  # type: ignore

    if backend == "mock":
        agent = InternVLAgent(
            backend="mock",
            model_id=model_id or "mock",
            tools=tools,
            generator_fn=kwargs.pop("generator_fn", None) or make_mock_generator(),
            device=device,
            auto_execute_tools=kwargs.pop("auto_execute_tools", True),
            use_vllm=use_vllm,
            **kwargs,
        )
        if auto_load:
            agent.load()
        return agent

    if backend == "smolvlm":
        agent = InternVLAgent(
            backend="smolvlm",
            model_id=model_id or DEFAULT_SMOLVLM_ID,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            torch_dtype=kwargs.pop("torch_dtype", "float16"),
            tools=tools,
            use_format_enforcer=kwargs.pop("use_format_enforcer", False),
            auto_execute_tools=kwargs.pop("auto_execute_tools", True),
            max_new_tokens=kwargs.pop("max_new_tokens", 128),
            use_vllm=use_vllm,
            **kwargs,
        )
        if auto_load:
            agent.load()
        return agent

    if backend in ("internvl2", "internvl"):
        agent = InternVLAgent(
            backend="internvl2",
            model_id=model_id or DEFAULT_INTERNVL_ID,
            device=device,
            load_in_8bit=load_in_8bit if load_in_8bit or load_in_4bit else True,
            load_in_4bit=load_in_4bit,
            torch_dtype=kwargs.pop("torch_dtype", "bfloat16"),
            tools=tools,
            use_format_enforcer=kwargs.pop("use_format_enforcer", True),
            auto_execute_tools=kwargs.pop("auto_execute_tools", True),
            use_vllm=use_vllm,
            **kwargs,
        )
        if auto_load:
            agent.load()
        return agent


    raise ValueError(
        f"Bilinmeyen VLM backend: {backend!r}. "
        f"Kullan: smolvlm | internvl2 | mock"
    )
