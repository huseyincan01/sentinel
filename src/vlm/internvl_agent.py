"""
VLM ajanı (SmolVLM / InternVL2 / mock) - Basitleştirilmiş
"""

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from src.vlm.memory import AgentMemory
from src.vlm.schemas import AnalysisResult
from src.vlm.tools import ToolRegistry, ToolResult

logger = logging.getLogger("sentinel.vlm")

DEFAULT_MODEL_ID = "OpenGVLab/InternVL2-8B"
DEFAULT_SMOLVLM_ID = "HuggingFaceTB/SmolVLM-Instruct"

_GLOBAL_MODEL_CACHE = {}

SYSTEM_PROMPT = (
    "Sen Sentinel adlı gelişmiş endüstriyel güvenlik ve tehlike analiz uzmanısın. "
    "Görüntüde ne olduğunu (alev, duman, yangın, devrilen forklift, yerde hareketsiz yatan insan vb.) "
    "açıkla. Yanıtın SADECE geçerli JSON olmalıdır."
)

def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try: return json.loads(text)
    except: pass
    
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("JSON bulunamadı")

class InternVLAgent:
    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "cuda",
        tools: Optional[ToolRegistry] = None,
        generator_fn: Optional[Callable[..., str]] = None,
        backend: str = "smolvlm",
        **kwargs
    ):
        self.backend = backend.lower()
        self.model_id = model_id or (DEFAULT_SMOLVLM_ID if self.backend == "smolvlm" else DEFAULT_MODEL_ID)
        self.device = device
        self.memory = AgentMemory(window_size=10, sticky_size=5)
        self.tools = tools or ToolRegistry()
        self.generator_fn = generator_fn
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._loaded = False
        self.last_tool_results: List[ToolResult] = []
        self._infer_lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._loaded or self.generator_fn is not None

    def load(self) -> "InternVLAgent":
        if self.backend == "mock" or self.generator_fn:
            self._loaded = True
            return self

        global _GLOBAL_MODEL_CACHE
        if self.model_id in _GLOBAL_MODEL_CACHE:
            cached = _GLOBAL_MODEL_CACHE[self.model_id]
            self.model, self.processor, self.tokenizer = cached["model"], cached.get("processor"), cached.get("tokenizer")
            self._loaded = True
            return self

        import torch
        if self.backend == "smolvlm":
            from transformers import AutoProcessor, AutoModelForVision2Seq
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
        else:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
        
        if torch.cuda.is_available() and self.device == "cuda":
            self.model = self.model.to("cuda")
        self.model.eval()
        
        _GLOBAL_MODEL_CACHE[self.model_id] = {"model": self.model, "processor": self.processor, "tokenizer": self.tokenizer}
        self._loaded = True
        return self

    def _prepare_image(self, image: Any):
        if image is None: return None
        from PIL import Image
        if isinstance(image, np.ndarray):
            rgb = image[:, :, ::-1].copy() if image.ndim == 3 and image.shape[2] == 3 else image
            return Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        return image.convert("RGB") if hasattr(image, "convert") else None

    def analyze_detail(self, frame: Any = None, frame_idx: int = 0, fps: float = 30.0, tracks: Optional[Sequence[Any]] = None, trigger_info: str = "", execute_tools: bool = True, **kwargs) -> AnalysisResult:
        track_str = ", ".join(str(getattr(t, "track_id", t)) for t in (tracks or []))
        prompt = f"{SYSTEM_PROMPT}\nZaman: {frame_idx/fps:.1f}s. Tracks: {track_str}. Info: {trigger_info}\nBeklenen JSON formatı:\n{{\"summary\":\"...\", \"events\":[], \"risk\":\"Orta\", \"risk_score\":0.5, \"actions\":[], \"tools_called\":[]}}"
        
        if self.generator_fn:
            raw = self.generator_fn(prompt=prompt, image=frame)
        else:
            with self._infer_lock:
                import torch
                pil = self._prepare_image(frame)
                
                if self.backend == "smolvlm":
                    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}] if pil else [{"type": "text", "text": prompt}]}]
                    text_in = self.processor.apply_chat_template(msgs, add_generation_prompt=True) + "\n```json\n{"
                    inputs = self.processor(text=text_in, images=[pil] if pil else None, return_tensors="pt").to(self.model.device)
                    with torch.no_grad():
                        out = self.model.generate(**inputs, max_new_tokens=512, repetition_penalty=1.3)
                    raw = "{" + self.processor.batch_decode(out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
                else:
                    text_in = f"<img></img>{prompt}" if pil else prompt
                    inputs = self.tokenizer(text_in, return_tensors="pt").to(self.model.device)
                    # Basit implementasyon
                    with torch.no_grad():
                        out = self.model.generate(**inputs, max_new_tokens=512)
                    raw = self.tokenizer.decode(out[0], skip_special_tokens=True)

        try:
            data = extract_json_object(raw)
            data.setdefault("risk", "Orta")
            data.setdefault("risk_score", 0.5)
            data.setdefault("summary", "Olay tespiti yapıldı.")
            data.setdefault("events", [])
            data.setdefault("actions", [])
            data.setdefault("tools_called", [])
            data.setdefault("frame_analyzed", frame_idx)
            result = AnalysisResult.model_validate(data)
        except Exception as e:
            logger.warning(f"JSON Parse Hatası: {e}")
            result = AnalysisResult(summary="Tehlike tespit edilemedi.", events=[], risk="Düşük", risk_score=0.1, actions=[], tools_called=[], frame_analyzed=frame_idx)
            
        self.memory.add(result)
        self.last_tool_results = []
        if execute_tools and result.tools_called:
            self.last_tool_results = self.tools.execute_from_analysis(result.tools_called, report_data=result.model_dump_report())
            
        return result

    def reset_memory(self):
        self.memory.reset()

def make_mock_generator(fixed: Optional[Dict[str, Any]] = None) -> Callable[..., str]:
    def _gen(prompt: str = "", image: Any = None, **kwargs: Any) -> str:
        if fixed: return json.dumps(fixed, ensure_ascii=False)
        p = prompt.lower()
        if "motion" in p or "roi" in p:
            return json.dumps({
                "summary": "Hareket veya tehlike algılandı.",
                "events": [{"time": "00:00", "event": "Hareket", "severity": "Yüksek"}],
                "risk": "Yüksek", "risk_score": 0.8,
                "actions": ["Güvenliği sağla"], "tools_called": ["trigger_alarm"],
                "frame_analyzed": 0
            }, ensure_ascii=False)
        return json.dumps({
            "summary": "Sakin görünüyor.", "events": [], "risk": "Düşük", "risk_score": 0.1, "actions": [], "tools_called": [], "frame_analyzed": 0
        }, ensure_ascii=False)
    return _gen
