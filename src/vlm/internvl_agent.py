"""
VLM ajanı (SmolVLM / InternVL2 / mock) - Ultra Hızlı Skor Sürümü
"""

import json
import logging
import re
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
    "Görüntüyü analiz et ve tehlike durumuna göre SADECE 0.0 ile 1.0 arasında bir risk skoru yaz. "
    "0.0 tamamen güvenli, 1.0 kritik kaza demektir. "
    "Başka HİÇBİR açıklama, kelime, harf veya JSON formatı KULLANMA. Sadece sayıyı yaz."
)

def extract_float_score(text: str) -> float:
    # Tüm metni temizleyip içindeki ilk ondalıklı sayıyı bulur
    matches = re.findall(r"0\.\d+|1\.0|0|1", text.strip())
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            pass
    return 0.1 # Fallback

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
        prompt = SYSTEM_PROMPT
        
        if self.generator_fn:
            raw = self.generator_fn(prompt=trigger_info, image=frame)
        else:
            with self._infer_lock:
                import torch
                pil = self._prepare_image(frame)
                
                if self.backend == "smolvlm":
                    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}] if pil else [{"type": "text", "text": prompt}]}]
                    text_in = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
                    inputs = self.processor(text=text_in, images=[pil] if pil else None, return_tensors="pt").to(self.model.device)
                    with torch.no_grad():
                        out = self.model.generate(**inputs, max_new_tokens=5, repetition_penalty=1.0)
                    raw = self.processor.batch_decode(out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
                else:
                    text_in = f"<img></img>{prompt}" if pil else prompt
                    inputs = self.tokenizer(text_in, return_tensors="pt").to(self.model.device)
                    with torch.no_grad():
                        out = self.model.generate(**inputs, max_new_tokens=5)
                    raw = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Skoru parse et
        score = extract_float_score(raw)
        
        # Sistemi kırmamak için AnalysisResult objesini manuel (sahte) oluşturuyoruz
        if score > 0.7:
            risk = "Yüksek"
            summary = f"Tehlike Algılandı (Skor: {score:.2f})"
        elif score > 0.4:
            risk = "Orta"
            summary = f"Şüpheli Durum (Skor: {score:.2f})"
        else:
            risk = "Düşük"
            summary = f"Güvenli (Skor: {score:.2f})"
            
        result = AnalysisResult(
            summary=summary,
            events=[],
            risk=risk,
            risk_score=score,
            actions=[],
            tools_called=[],
            frame_analyzed=frame_idx
        )
            
        self.memory.add(result)
        self.last_tool_results = []
        
        # Araç çalıştırmayı kapattık çünkü model artık tool listesi üretmiyor.
        
        return result

    def reset_memory(self):
        self.memory.reset()

def make_mock_generator(fixed: Optional[Dict[str, Any]] = None) -> Callable[..., str]:
    def _gen(prompt: str = "", image: Any = None, **kwargs: Any) -> str:
        if fixed: return str(fixed.get("risk_score", 0.5))
        p = prompt.lower()
        if "motion" in p or "roi" in p:
            return "0.85"
        return "0.1"
    return _gen
