"""
VLM ajanı (SmolVLM / InternVL2 / mock).

Pipeline tek yol kullanır: analyze_detail → AnalysisResult + tools + memory.
Girdi boyutu pipeline'da 336×336'ya sabitlenir (high-res Detail yolu yok).

analyze_gate / GateDecision: eski API; pipeline kullanmaz (test uyumu).
HF generate() + opsiyonel lm-format-enforcer; chat() kullanılmaz.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from pydantic import ValidationError

from src.vlm.memory import AgentMemory
from src.vlm.schemas import (
    AnalysisResult,
    GateDecision,
    KNOWN_TOOLS,
    analysis_json_schema,
    gate_json_schema,
)
from src.vlm.tools import ToolRegistry, ToolResult

logger = logging.getLogger("sentinel.vlm")

DEFAULT_MODEL_ID = "OpenGVLab/InternVL2-8B"
DEFAULT_SMOLVLM_ID = "HuggingFaceTB/SmolVLM-Instruct"
DEFAULT_BACKEND = "smolvlm"  # yerel geliştirme varsayılanı

# Küresel model önbelleği (RAM/VRAM tasarrufu ve hızlı başlatma için)
_GLOBAL_MODEL_CACHE = {}

SYSTEM_PROMPT_DETAIL = (
    "Sen Sentinel adlı gelişmiş endüstriyel güvenlik ve tehlike analiz uzmanısın. "
    "YÜKSEK çözünürlüklü (yakınlaştırılmış/odaklanmış) görüntüyü ve geçmiş olay bağlamını incele. "
    "Görüntüde ne olduğunu (örneğin: alev, duman, yangın, kıvılcım, devrilen forklift, yerde hareketsiz yatan insan, "
    "baret/yelek takmayan personel, tehlikeli iş aletleri vb.) fiziksel ve görsel detaylarıyla tam olarak açıkla. "
    "Yanıtın SADECE geçerli JSON olmalıdır. Türkçe 'summary' alanında olayı detaylıca açıkla."
)

SYSTEM_PROMPT_GATE = (
    "Sen Sentinel güvenlik hakemisin. DÜŞÜK çözünürlüklü görüntü ve aday sinyaller verildi. "
    "Görüntüde herhangi bir yangın (alev/duman), kaza, devrilme, yaralanma veya baret/yelek ihlali şüphesi var mı? "
    "Görevin: yüksek çözünürlüklü detaylı VLM analizi GEREKLİ Mİ karar ver. "
    "Yanıtın SADECE şu alanlarda JSON: need_high_res (bool), confidence (0-1), "
    "reason (Türkçe gerekçe), urgency (low|medium|high|critical)."
)


def _frame_time_str(frame_idx: int, fps: float = 30.0) -> str:
    total_sec = int(frame_idx / max(fps, 1e-6))
    mm, ss = divmod(total_sec, 60)
    return f"{mm:02d}:{ss:02d}"


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"JSON nesnesi bulunamadı: {text[:200]!r}")


def build_analysis_prompt(
    memory_context: str,
    frame_idx: int = 0,
    fps: float = 30.0,
    track_context: str = "",
    extra: str = "",
) -> str:
    t = _frame_time_str(frame_idx, fps)
    parts = [
        SYSTEM_PROMPT_DETAIL,
        "",
        memory_context,
        "",
        f"Şu anki kare zamanı: {t} (frame={frame_idx}).",
    ]
    if track_context:
        parts.append(f"Takip bağlamı: {track_context}")
    if extra:
        parts.append(extra)
    parts.append("Şimdi bu kareyi analiz et ve yalnızca JSON döndür:")
    return "\n".join(parts)


def build_gate_prompt(
    candidate_summary: str,
    frame_idx: int = 0,
    fps: float = 30.0,
    track_context: str = "",
) -> str:
    t = _frame_time_str(frame_idx, fps)
    parts = [
        SYSTEM_PROMPT_GATE,
        "",
        f"Zaman: {t} (frame={frame_idx}).",
        f"Aday sinyaller: {candidate_summary or 'yok'}",
    ]
    if track_context:
        parts.append(f"Track: {track_context}")
    parts.append(
        "Bu düşük çözünürlüklü kareye bakarak high-res detay gerekli mi? "
        "Yalnızca Gate JSON döndür."
    )
    return "\n".join(parts)


class InternVLAgent:
    """
    Gate (low-res) + Detail (high-res) VLM ajanı.

    backend:
      - smolvlm  : geliştirme (4050 varsayılan)
      - internvl2: hedef / Colab
      - mock     : generator_fn
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: str = "bfloat16",
        window_size: int = 10,
        sticky_size: int = 5,
        tools: Optional[ToolRegistry] = None,
        generator_fn: Optional[Callable[..., str]] = None,
        gate_generator_fn: Optional[Callable[..., str]] = None,
        use_format_enforcer: bool = True,
        max_new_tokens: int = 512,
        max_gate_tokens: int = 128,
        auto_execute_tools: bool = True,
        backend: str = DEFAULT_BACKEND,
        use_vllm: bool = False,
    ) -> None:
        self.backend = (backend or DEFAULT_BACKEND).lower().strip()
        if self.backend in ("internvl", "internvl2-8b"):
            self.backend = "internvl2"
        if model_id is None:
            if self.backend == "smolvlm":
                model_id = DEFAULT_SMOLVLM_ID
            elif self.backend == "mock":
                model_id = "mock"
            else:
                model_id = DEFAULT_MODEL_ID
        self.model_id = model_id
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        # SmolVLM için float16 genelde daha uyumlu
        if self.backend == "smolvlm" and torch_dtype == "bfloat16":
            torch_dtype = "float16"
        self.torch_dtype = torch_dtype
        self.memory = AgentMemory(window_size=window_size, sticky_size=sticky_size)
        self.tools = tools or ToolRegistry()
        self.generator_fn = generator_fn
        self.gate_generator_fn = gate_generator_fn
        # SmolVLM'de lm-format-enforcer tokenizer uyumu zayıf olabilir
        if self.backend == "smolvlm":
            use_format_enforcer = False
        self.use_format_enforcer = use_format_enforcer
        self.max_new_tokens = max_new_tokens
        self.max_gate_tokens = max_gate_tokens
        self.auto_execute_tools = auto_execute_tools
        self.use_vllm = use_vllm

        self.model = None
        self.tokenizer = None
        self.processor = None  # SmolVLM
        self.vllm_llm = None  # vLLM motoru
        self._prefix_fn = None
        self._gate_prefix_fn = None
        self._loaded = False
        self.last_raw_text: Optional[str] = None
        self.last_gate_raw: Optional[str] = None
        self.last_tool_results: List[ToolResult] = []
        self.last_gate: Optional[GateDecision] = None
        # Gate/detail aynı GPU modelini paylaşır → generate serileşir
        # Gate asla detail bitene kadar BEKLEMEZ (try_acquire); periodu kaçırır.
        self._infer_lock = threading.RLock()


    def try_begin_infer(self, blocking: bool = False, timeout: float = -1) -> bool:
        """Model kilidini al. Gate için blocking=False kullan."""
        if blocking:
            if timeout is not None and timeout >= 0:
                return self._infer_lock.acquire(blocking=True, timeout=timeout)
            return self._infer_lock.acquire(blocking=True)
        return self._infer_lock.acquire(blocking=False)

    def end_infer(self) -> None:
        try:
            self._infer_lock.release()
        except RuntimeError:
            pass

    @property
    def is_loaded(self) -> bool:
        return self._loaded or self.generator_fn is not None or self.gate_generator_fn is not None

    def load(self) -> "InternVLAgent":
        if self.backend == "mock" or self.generator_fn is not None or self.gate_generator_fn is not None:
            self._loaded = True
            logger.info("Mock / generator aktif; ağır model indirilmedi. backend=%s", self.backend)
            return self

        # vLLM motoru aktifse önce vLLM'i dene
        if self.use_vllm:
            try:
                from vllm import LLM
                logger.info("vLLM motoru ile yükleniyor: %s", self.model_id)
                self.vllm_llm = LLM(
                    model=self.model_id,
                    trust_remote_code=True,
                    max_model_len=2048 if self.backend == "smolvlm" else 4096,
                    gpu_memory_utilization=0.80,  # YOLO ve MOG2 için VRAM payı bırak
                    limit_mm_per_prompt={"image": 1},
                )
                self._loaded = True
                logger.info("vLLM motoru başarıyla yüklendi.")
                return self
            except Exception as exc:
                logger.warning("vLLM motoru yüklenemedi (büyük ihtimalle Windows veya paket eksik), Transformers motoruna geri dönülüyor: %s", exc)
                self.use_vllm = False

        # Küresel önbellekten kontrol et (transformers için)
        cache_key = (self.backend, self.model_id, self.torch_dtype, self.load_in_4bit, self.load_in_8bit)
        global _GLOBAL_MODEL_CACHE
        if cache_key in _GLOBAL_MODEL_CACHE:
            cached = _GLOBAL_MODEL_CACHE[cache_key]
            self.model = cached["model"]
            self.processor = cached.get("processor")
            self.tokenizer = cached.get("tokenizer")
            self._loaded = True
            logger.info("Model küresel önbellekten (VRAM) hızlıca alındı. backend=%s", self.backend)
            return self

        if self.backend == "smolvlm":
            res = self._load_smolvlm()
        else:
            res = self._load_internvl()

        # Sonucu küresel önbelleğe yaz
        _GLOBAL_MODEL_CACHE[cache_key] = {
            "model": self.model,
            "processor": self.processor,
            "tokenizer": self.tokenizer,
        }
        return res


    def _dtype(self):
        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype.lower(), torch.float16)

    def _load_smolvlm(self) -> "InternVLAgent":
        """HuggingFace SmolVLM — 4050 geliştirme varsayılanı."""
        import torch
        from transformers import AutoProcessor

        dtype = self._dtype()
        logger.info("SmolVLM yükleniyor: %s", self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if self.load_in_4bit or self.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig

                if self.load_in_4bit:
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                else:
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                load_kwargs["device_map"] = "auto"
            except Exception as exc:
                logger.warning("bitsandbytes yok/başarısız, fp16 denenecek: %s", exc)
                load_kwargs.pop("quantization_config", None)

        # Model sınıfı: Vision2Seq veya Idefics3
        model = None
        err_last = None
        for loader_name in (
            "AutoModelForVision2Seq",
            "AutoModelForImageTextToText",
            "Idefics3ForConditionalGeneration",
            "AutoModel",
        ):
            try:
                mod = __import__("transformers", fromlist=[loader_name])
                cls = getattr(mod, loader_name, None)
                if cls is None:
                    continue
                kw = dict(load_kwargs)
                if loader_name == "AutoModel":
                    kw["trust_remote_code"] = True
                model = cls.from_pretrained(self.model_id, **kw)
                logger.info("SmolVLM yüklendi via %s", loader_name)
                break
            except Exception as exc:
                err_last = exc
                continue
        if model is None:
            raise RuntimeError(f"SmolVLM yüklenemedi: {err_last}")

        if "device_map" not in load_kwargs:
            if self.device == "cuda" and torch.cuda.is_available():
                model = model.to("cuda")
            else:
                model = model.to("cpu")
                self.device = "cpu"
        model.eval()
        self.model = model
        # tokenizer alias
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        self._loaded = True
        logger.info("SmolVLM hazır (geliştirme backend).")
        return self

    def _load_internvl(self) -> "InternVLAgent":
        import torch
        from transformers import AutoModel, AutoTokenizer

        dtype = self._dtype()
        logger.info("InternVL yükleniyor: %s", self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False
        )
        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs["device_map"] = "auto"
            except Exception as exc:
                logger.warning("4-bit başarısız: %s", exc)
                load_kwargs["torch_dtype"] = dtype
        elif self.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                load_kwargs["device_map"] = "auto"
            except Exception as exc:
                logger.warning("8-bit başarısız: %s", exc)
                load_kwargs["torch_dtype"] = dtype
        else:
            load_kwargs["torch_dtype"] = dtype
            if self.device == "cuda" and torch.cuda.is_available():
                load_kwargs["device_map"] = "auto"

        self.model = AutoModel.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()
        if self.use_format_enforcer:
            self._prefix_fn = self._build_prefix_fn(analysis_json_schema())
            self._gate_prefix_fn = self._build_prefix_fn(gate_json_schema())
        self._loaded = True
        return self

    def _build_prefix_fn(self, schema: dict) -> Optional[Callable]:
        if self.tokenizer is None:
            return None
        try:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import (
                build_transformers_prefix_allowed_tokens_fn,
            )

            return build_transformers_prefix_allowed_tokens_fn(
                self.tokenizer, JsonSchemaParser(schema)
            )
        except Exception as exc:
            logger.warning("prefix_fn kurulamadı: %s", exc)
            return None

    def build_prompt(
        self,
        frame_idx: int = 0,
        fps: float = 30.0,
        track_context: str = "",
        extra: str = "",
    ) -> str:
        prompt = build_analysis_prompt(
            memory_context=self.memory.build_context_prompt(),
            frame_idx=frame_idx,
            fps=fps,
            track_context=track_context,
            extra=extra,
        )
        if not self.use_format_enforcer:
            prompt += (
                "\n\nYanıtını kesinlikle aşağıdaki JSON şemasına uygun olarak üretmelisin. "
                "Cevabında JSON bloğu dışında hiçbir açıklama veya ek metin bulunmamalıdır:\n"
                "{\n"
                '  "summary": "olayın Türkçe kısa açıklaması (str)",\n'
                '  "events": [\n'
                '    {"time": "00:00", "event": "olay açıklaması (str)", "severity": "Düşük/Orta/Yüksek/Kritik"}\n'
                '  ],\n'
                '  "risk": "Düşük/Orta/Yüksek/Kritik",\n'
                '  "risk_score": 0.5,\n'
                '  "actions": ["aksiyon 1", "aksiyon 2"],\n'
                '  "tools_called": ["call_ambulance", "lock_area"],\n'
                '  "timestamp": "2026-07-15T00:10:00",\n'
                '  "frame_analyzed": 0\n'
                "}\n"
            )
        return prompt

    def _generate_text(
        self,
        prompt: str,
        image: Any = None,
        max_new_tokens: Optional[int] = None,
        prefix_fn: Any = None,
        use_gate_generator: bool = False,
    ) -> str:
        # Mock/generator yolları kilitsiz (test hızı); gerçek model.generate seri
        if use_gate_generator and self.gate_generator_fn is not None:
            return self.gate_generator_fn(prompt=prompt, image=image)
        if self.generator_fn is not None and not use_gate_generator:
            return self.generator_fn(prompt=prompt, image=image)
        if use_gate_generator and self.gate_generator_fn is None and self.generator_fn is not None:
            return make_mock_gate_generator()(prompt=prompt, image=image)

        # Dışarıdan try_begin_infer ile kilit alınmış olabilir (reentrant RLock)
        with self._infer_lock:
            return self._generate_text_locked(
                prompt, image, max_new_tokens, prefix_fn
            )

    def _generate_text_locked(
        self,
        prompt: str,
        image: Any = None,
        max_new_tokens: Optional[int] = None,
        prefix_fn: Any = None,
    ) -> str:
        if self.use_vllm and self.vllm_llm is not None:
            return self._generate_vllm(prompt, image, max_new_tokens)

        if self.backend == "smolvlm" and self.model is not None and self.processor is not None:
            return self._generate_smolvlm(prompt, image, max_new_tokens)

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model yüklü değil. load() veya generator_fn verin. (Veya vLLM yüklenemedi)")


        import torch

        pixel_values = self._prepare_image(image) if image is not None else None
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        if hasattr(self.model, "device"):
            input_ids = input_ids.to(self.model.device)
        elif torch.cuda.is_available() and self.device == "cuda":
            input_ids = input_ids.cuda()

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": False,
            "repetition_penalty": 1.3,
        }
        pfn = prefix_fn if prefix_fn is not None else self._prefix_fn
        if pfn is not None:
            gen_kwargs["prefix_allowed_tokens_fn"] = pfn

        with torch.no_grad():
            try:
                if pixel_values is not None:
                    output_ids = self.model.generate(
                        input_ids, pixel_values=pixel_values, **gen_kwargs
                    )
                else:
                    output_ids = self.model.generate(input_ids, **gen_kwargs)
            except TypeError:
                output_ids = self.model.generate(input_ids, **gen_kwargs)

        gen_ids = output_ids[0]
        if gen_ids.shape[0] > input_ids.shape[-1]:
            gen_ids = gen_ids[input_ids.shape[-1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def _numpy_to_pil(self, image: Any):
        from PIL import Image

        if image is None:
            return None
        if hasattr(image, "convert"):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = arr[:, :, ::-1].copy()  # BGR→RGB
            return Image.fromarray(np.asarray(arr, dtype=np.uint8))
        return None

    def _generate_vllm(
        self,
        prompt: str,
        image: Any = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """vLLM vision-language model generation logic."""
        from vllm import SamplingParams

        pil = self._numpy_to_pil(image)

        # Prompt formatting
        if self.backend == "smolvlm":
            if self.processor is None:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(self.model_id)

            messages = [
                {
                    "role": "user",
                    "content": (
                        [{"type": "image"}, {"type": "text", "text": prompt}]
                        if pil is not None
                        else [{"type": "text", "text": prompt}]
                    ),
                }
            ]
            try:
                text_in = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
            except Exception:
                text_in = prompt
            text_in = text_in.rstrip() + "\n```json\n{"
        elif self.backend == "internvl2":
            if self.tokenizer is None:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            text_in = f"<img></img>{prompt}" if pil is not None else prompt
        else:
            text_in = prompt

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens or self.max_new_tokens,
            temperature=0.0,
            repetition_penalty=1.3,
        )

        if pil is not None:
            inputs = {
                "prompt": text_in,
                "multi_modal_data": {"image": pil}
            }
        else:
            inputs = {
                "prompt": text_in
            }

        outputs = self.vllm_llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
        text = outputs[0].outputs[0].text

        if self.backend == "smolvlm":
            text = "{\n" + text.strip()
            text = text.split("```")[0].strip()

        return text

    def _generate_smolvlm(

        self,
        prompt: str,
        image: Any = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """SmolVLM processor + generate."""
        import torch

        pil = self._numpy_to_pil(image)
        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image"}, {"type": "text", "text": prompt}]
                    if pil is not None
                    else [{"type": "text", "text": prompt}]
                ),
            }
        ]
        try:
            text_in = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
        except Exception:
            text_in = prompt

        # JSON'a zorlamak için modele cevaba '{' ile başlamasını dikte et
        text_in = text_in.rstrip() + "\n```json\n{"

        if pil is not None:
            inputs = self.processor(
                text=text_in,
                images=[pil],
                return_tensors="pt",
            )
        else:
            inputs = self.processor(text=text_in, return_tensors="pt")

        # Cihaza taşı
        move_to = self.device if self.device == "cpu" else "cuda"
        if move_to == "cuda" and not torch.cuda.is_available():
            move_to = "cpu"
        inputs = {
            k: v.to(move_to) if hasattr(v, "to") else v for k, v in inputs.items()
        }

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.3,
            )
        # Girdi uzunluğunu at
        in_len = inputs["input_ids"].shape[-1]
        out_ids = generated[:, in_len:]
        if self.processor is not None and hasattr(self.processor, "batch_decode"):
            text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        elif self.tokenizer is not None:
            text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        else:
            text = str(out_ids)
        
        # Enjekte ettiğimiz süslü parantezi geri ekle ve markdown sonunu temizle
        text = "{\n" + text.strip()
        text = text.split("```")[0].strip()
        return text

    def _prepare_image(self, image: Any):
        """BGR/RGB veya PIL → tensör (best-effort). Always-true koşullar yok."""
        try:
            import torch
            from PIL import Image

            if isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[2] == 3:
                    rgb = image[:, :, ::-1].copy()
                else:
                    rgb = image
                pil = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
            elif hasattr(image, "convert"):
                pil = image.convert("RGB")
            else:
                return None

            self._last_pil_image = pil

            if self.model is not None and hasattr(self.model, "vision_model"):
                try:
                    arr = np.asarray(pil.resize((448, 448)), dtype=np.float32) / 255.0
                    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                    if self.device == "cuda" and torch.cuda.is_available():
                        tensor = tensor.cuda()
                    return tensor
                except Exception as exc:
                    logger.debug("Vision tensör hazırlama: %s", exc)
                    return None
            return None
        except Exception as exc:
            logger.debug("Görüntü hazırlama: %s", exc)
            return None

    def parse_and_validate(
        self,
        text: str,
        frame_idx: int = 0,
        default_time: Optional[str] = None,
    ) -> AnalysisResult:
        self.last_raw_text = text
        try:
            data = extract_json_object(text)
            
            # Defansif alan atamaları (şema uyumunu garantilemek için)
            if not isinstance(data.get("events"), list):
                data["events"] = []
            if not isinstance(data.get("actions"), list):
                if isinstance(data.get("actions"), str):
                    data["actions"] = [data["actions"]]
                else:
                    data["actions"] = []
            if not isinstance(data.get("tools_called"), list):
                if isinstance(data.get("tools_called"), str):
                    data["tools_called"] = [data["tools_called"]]
                else:
                    data["tools_called"] = []
            if "risk" not in data:
                data["risk"] = "Orta"
            if "risk_score" not in data:
                data["risk_score"] = 0.5
            if "summary" not in data or not data["summary"]:
                data["summary"] = "Olay tespiti yapıldı (detaylar rapordadır)."

            data.setdefault("frame_analyzed", frame_idx)
            data.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

            if "events" in data and isinstance(data["events"], list):
                for ev in data["events"]:
                    if isinstance(ev, dict):
                        t_val = str(ev.get("time", "")).strip()
                        parts = t_val.split(":")
                        is_valid = True
                        if len(parts) not in (2, 3):
                            is_valid = False
                        else:
                            for p in parts:
                                if not p.isdigit():
                                    is_valid = False
                                    break
                        if not is_valid:
                            ev["time"] = default_time or "00:00"
            return AnalysisResult.model_validate(data)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Detail JSON fallback Hata: %s | Raw text: %r", exc, text)
            return AnalysisResult(
                summary="Tehlike tespit edilemedi.",
                events=[],
                risk="Düşük",
                risk_score=0.1,
                actions=["İzlemeye devam et"],
                tools_called=[],
                frame_analyzed=frame_idx,
            )

    def parse_gate(self, text: str) -> GateDecision:
        self.last_gate_raw = text
        try:
            data = extract_json_object(text)
            return GateDecision.model_validate(data)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Gate JSON fallback: %s | Raw text: %r", exc, text)
            return GateDecision(
                need_high_res=False,
                confidence=0.0,
                reason="Analiz edilemedi",
                urgency="low",
            )

    def analyze_gate(
        self,
        frame: Any = None,
        frame_idx: int = 0,
        fps: float = 30.0,
        tracks: Optional[Sequence[Any]] = None,
        candidate_summary: str = "",
    ) -> GateDecision:
        """
        Low-res hakem: high-res detail gerekli mi?

        frame tercihen zaten düşük çözünürlüklü olmalı (pipeline downscale eder).
        """
        track_context = self._format_tracks(tracks)
        prompt = build_gate_prompt(
            candidate_summary=candidate_summary,
            frame_idx=frame_idx,
            fps=fps,
            track_context=track_context,
        )
        if not self.use_format_enforcer:
            prompt += (
                "\n\nYanıtını kesinlikle aşağıdaki JSON şemasına uygun olarak üretmelisin. "
                "Cevabında JSON bloğu dışında hiçbir açıklama veya ek metin bulunmamalıdır:\n"
                "{\n"
                '  "need_high_res": true/false,\n'
                '  "confidence": 0.8,\n'
                '  "reason": "kısa Türkçe gerekçe",\n'
                '  "urgency": "low/medium/high/critical"\n'
                "}\n"
            )
        raw = self._generate_text(
            prompt,
            image=frame,
            max_new_tokens=self.max_gate_tokens,
            prefix_fn=self._gate_prefix_fn,
            use_gate_generator=True,
        )
        gate = self.parse_gate(raw)
        self.last_gate = gate
        return gate

    def analyze(
        self,
        frame: Any = None,
        frame_idx: int = 0,
        fps: float = 30.0,
        tracks: Optional[Sequence[Any]] = None,
        trigger_info: str = "",
        execute_tools: Optional[bool] = None,
    ) -> AnalysisResult:
        """Detail analiz (high-res) — analyze_detail ile aynı."""
        return self.analyze_detail(
            frame=frame,
            frame_idx=frame_idx,
            fps=fps,
            tracks=tracks,
            trigger_info=trigger_info,
            execute_tools=execute_tools,
        )

    def analyze_detail(
        self,
        frame: Any = None,
        frame_idx: int = 0,
        fps: float = 30.0,
        tracks: Optional[Sequence[Any]] = None,
        trigger_info: str = "",
        execute_tools: Optional[bool] = None,
    ) -> AnalysisResult:
        """High-res detaylı analiz + memory + tools."""
        track_context = self._format_tracks(tracks)
        extra = f"Tetikleyici/geçit: {trigger_info}" if trigger_info else ""
        prompt = self.build_prompt(
            frame_idx=frame_idx,
            fps=fps,
            track_context=track_context,
            extra=extra,
        )
        raw = self._generate_text(
            prompt,
            image=frame,
            max_new_tokens=self.max_new_tokens,
            prefix_fn=self._prefix_fn,
            use_gate_generator=False,
        )
        result = self.parse_and_validate(
            raw,
            frame_idx=frame_idx,
            default_time=_frame_time_str(frame_idx, fps),
        )
        result.frame_analyzed = frame_idx
        self.memory.add(result)

        do_tools = self.auto_execute_tools if execute_tools is None else execute_tools
        self.last_tool_results = []
        if do_tools and result.tools_called:
            self.last_tool_results = self.tools.execute_from_analysis(
                result.tools_called,
                report_data=result.model_dump_report(),
            )
        return result

    @staticmethod
    def _format_tracks(tracks: Optional[Sequence[Any]]) -> str:
        if not tracks:
            return ""
        parts = []
        for tr in tracks:
            if isinstance(tr, dict):
                tid = tr.get("track_id", "?")
                src = tr.get("source", "")
                parts.append(f"{tid}({src})" if src else str(tid))
            else:
                tid = getattr(tr, "track_id", None)
                src = getattr(tr, "source", "")
                if tid is not None:
                    parts.append(f"{tid}({src})" if src else str(tid))
        return ", ".join(parts)

    def get_memory_prompt(self) -> str:
        return self.memory.build_context_prompt()

    def reset_memory(self) -> None:
        self.memory.reset()

    def build_format_enforcer_parser(self):
        try:
            from lmformatenforcer import JsonSchemaParser

            return JsonSchemaParser(analysis_json_schema())
        except Exception as exc:
            logger.warning("JsonSchemaParser: %s", exc)
            return None


def make_mock_generator(fixed: Optional[Dict[str, Any]] = None) -> Callable[..., str]:
    """Detail VLM mock."""

    def _gen(prompt: str = "", image: Any = None, **kwargs: Any) -> str:
        del image, kwargs
        if fixed is not None:
            return json.dumps(fixed, ensure_ascii=False)
        low = {
            "summary": "Sahne sakin görünüyor; belirgin tehlike yok.",
            "events": [
                {"time": "00:00", "event": "Rutin gözlem", "severity": "Düşük"}
            ],
            "risk": "Düşük",
            "risk_score": 0.15,
            "actions": ["İzlemeye devam et"],
            "tools_called": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_analyzed": 0,
        }
        high = {
            "summary": "Videoda forklift kazası ve yaralanma riski gözlenmiştir.",
            "events": [
                {"time": "00:15", "event": "Forklift devrildi", "severity": "Yüksek"},
                {
                    "time": "00:20",
                    "event": "Yerde hareketsiz kişi",
                    "severity": "Kritik",
                },
            ],
            "risk": "Yüksek",
            "risk_score": 0.87,
            "actions": ["Sağlık ekibini çağır", "Alanı güvenlik altına al"],
            "tools_called": [
                "call_ambulance",
                "lock_area",
                "trigger_alarm",
                "generate_incident_report",
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_analyzed": 450,
        }
        p = (prompt or "").lower()
        # Hafıza/geçmiş kilitlenmesini önlemek için yalnızca şu anki kare tetiğine bak
        current_part = p
        if "[odak]" in p:
            current_part = p.split("[odak]")[-1]

        is_danger = any(k in current_part for k in ("stillness", "dangerous_motion", "color_fire", "roi", "entrance"))
        if is_danger:
            return json.dumps(high, ensure_ascii=False)
        return json.dumps(low, ensure_ascii=False)


    return _gen


def make_mock_gate_generator(
    fixed: Optional[Dict[str, Any]] = None,
) -> Callable[..., str]:
    """Gate VLM mock — yalnızca aday satırına göre need_high_res.

    Not: Tam prompt içinde SYSTEM_PROMPT_GATE 'kaza/devril' içerir; tüm prompt
    taranırsa mock her zaman escalate ederdi (periyodik gate'i detail'e boğardı).
    """

    def _gen(prompt: str = "", image: Any = None, **kwargs: Any) -> str:
        del image, kwargs
        if fixed is not None:
            return json.dumps(fixed, ensure_ascii=False)
        text = prompt or ""
        # Sadece aday özeti satırına bak (sistem promptundaki tehlike kelimelerini yoksay)
        cand_line = ""
        for line in text.splitlines():
            if line.lower().startswith("aday sinyaller:"):
                cand_line = line
                break
        p = cand_line.lower() if cand_line else ""
        # Tehlikeli aday türleri → high-res iste
        danger = any(
            k in p
            for k in (
                "stillness",
                "hareketsiz",
                "ssim",
                "şiddet",
                "siddet",
                "kaza",
                "devril",
                "critical",
                "kritik",
                "dangerous",
                "entrance",
                "giriş",
                "giris",
                "mog2",
                "color_fire",
                "yangın",
                "yangin",
                "fire",
            )
        )
        # "aday" / "yok" / "periodic" tek başına escalate etmez
        if danger:
            out = {
                "need_high_res": True,
                "confidence": 0.85,
                "reason": "Aday sinyaller detay analizi gerektiriyor",
                "urgency": "high",
            }
        else:
            out = {
                "need_high_res": False,
                "confidence": 0.7,
                "reason": "Rutin görünüm; high-res gerekmiyor",
                "urgency": "low",
            }
        return json.dumps(out, ensure_ascii=False)

    return _gen
