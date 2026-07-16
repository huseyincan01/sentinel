"""
Hibrit Tracker: YOLOv8 + MOG2 + BoT-SORT.

- YOLO tespitleri → BoT-SORT (ReID aktif) → track ID prefix: yolo_
- MOG2 blob'ları  → BoT-SORT (ReID kapalı) → track ID prefix: mog_

Namespace ayrımı, bellek/rapor katmanında ID çakışmasını önler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from src.tracking.mog2_detector import BlobDetection, MOG2Detector

# Proje köküne göre config yolları
_TRACKING_DIR = Path(__file__).resolve().parent
DEFAULT_MOG_TRACKER_CFG = _TRACKING_DIR / "botsort_config.yaml"
DEFAULT_YOLO_TRACKER_CFG = _TRACKING_DIR / "botsort_config_yolo.yaml"

YOLO_PREFIX = "yolo_"
MOG_PREFIX = "mog_"


@dataclass
class TrackedObject:
    """Tek bir kalıcı-ID'li takip nesnesi."""

    track_id: str  # örn. yolo_3 veya mog_1
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    source: str  # "yolo" | "mog2"
    frame_idx: int
    with_reid: bool  # Bu track hattında ReID kullanıldı mı?


@dataclass
class FrameTracks:
    """Bir karedeki tüm track sonuçları."""

    frame_idx: int
    tracks: List[TrackedObject] = field(default_factory=list)
    yolo_detections: int = 0
    mog2_detections: int = 0

    @property
    def yolo_tracks(self) -> List[TrackedObject]:
        return [t for t in self.tracks if t.source == "yolo"]

    @property
    def mog2_tracks(self) -> List[TrackedObject]:
        return [t for t in self.tracks if t.source == "mog2"]


def load_tracker_args(config_path: Union[str, Path]) -> SimpleNamespace:
    """YAML tracker config'ini ultralytics uyumlu SimpleNamespace'e çevir."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    # Ultralytics BOTSORT beklediği alanlar
    defaults = {
        "tracker_type": "botsort",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.25,
        "with_reid": False,
        "model": "auto",
    }
    defaults.update({k: v for k, v in raw.items() if not str(k).startswith("#")})
    return SimpleNamespace(**defaults)


def _try_create_botsort(args: SimpleNamespace, frame_rate: int = 30):
    """
    Ultralytics BOTSORT örneği oluştur.

    Sürüm farklarında import yolu değişebilir; başarısız olursa None döner
    ve basit IoU tracker'a düşülür.
    """
    try:
        from ultralytics.trackers.bot_sort import BOTSORT

        return BOTSORT(args=args, frame_rate=frame_rate)
    except Exception:
        try:
            from ultralytics.trackers.bot_sort import BoTSORT as BOTSORT

            return BOTSORT(args=args, frame_rate=frame_rate)
        except Exception:
            return None


class SimpleIoUTracker:
    """
    Hafif IoU + merkez mesafesi tracker (ReID YOK).

    BoT-SORT kullanılamadığında veya testlerde deterministik davranış için
    yedek. MOG2 için ideal: sadece geometrik eşleştirme.
    """

    def __init__(
        self,
        match_thresh: float = 0.3,
        max_age: int = 30,
        with_reid: bool = False,
    ) -> None:
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.with_reid = with_reid  # Her zaman False olmalı (MOG2)
        self._next_id = 1
        # id -> {bbox, conf, cls, age, hits}
        self._tracks: Dict[int, Dict[str, Any]] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = a[:4]
        bx1, by1, bx2, by2 = b[:4]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    def update(
        self, detections: np.ndarray, img: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Args:
            detections: (N, 5+) x1,y1,x2,y2,conf[,cls]
            img: kullanılmaz (ReID yok); imza uyumu için.

        Returns:
            (M, 7+) x1,y1,x2,y2,track_id,conf,cls  formatına yakın dizi.
        """
        del img  # ReID kullanılmıyor
        if detections is None or len(detections) == 0:
            detections = np.zeros((0, 5), dtype=np.float32)

        dets = np.asarray(detections, dtype=np.float32)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        # Yaşlandır
        for tid in list(self._tracks.keys()):
            self._tracks[tid]["age"] += 1

        track_ids = list(self._tracks.keys())
        matched_t: set = set()
        matched_d: set = set()
        pairs: List[Tuple[int, int, float]] = []

        for ti, tid in enumerate(track_ids):
            tb = self._tracks[tid]["bbox"]
            for di, det in enumerate(dets):
                iou = self._iou(tb, det)
                if iou >= self.match_thresh:
                    pairs.append((ti, di, iou))

        pairs.sort(key=lambda x: x[2], reverse=True)
        for ti, di, _ in pairs:
            if ti in matched_t or di in matched_d:
                continue
            matched_t.add(ti)
            matched_d.add(di)
            tid = track_ids[ti]
            det = dets[di]
            conf = float(det[4]) if det.shape[0] > 4 else 0.5
            cls = int(det[5]) if det.shape[0] > 5 else 0
            self._tracks[tid] = {
                "bbox": det[:4].copy(),
                "conf": conf,
                "cls": cls,
                "age": 0,
                "hits": self._tracks[tid]["hits"] + 1,
            }

        # Yeni track'ler
        for di, det in enumerate(dets):
            if di in matched_d:
                continue
            conf = float(det[4]) if det.shape[0] > 4 else 0.5
            cls = int(det[5]) if det.shape[0] > 5 else 0
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {
                "bbox": det[:4].copy(),
                "conf": conf,
                "cls": cls,
                "age": 0,
                "hits": 1,
            }

        # Eski track'leri sil
        for tid in list(self._tracks.keys()):
            if self._tracks[tid]["age"] > self.max_age:
                del self._tracks[tid]

        # Aktif (bu karede görülen) track'leri döndür
        results = []
        for tid, tr in self._tracks.items():
            if tr["age"] == 0:
                x1, y1, x2, y2 = tr["bbox"]
                results.append(
                    [x1, y1, x2, y2, float(tid), tr["conf"], float(tr["cls"])]
                )
        if not results:
            return np.zeros((0, 7), dtype=np.float32)
        return np.asarray(results, dtype=np.float32)


class HybridTracker:
    """
    Ana hibrit takip sınıfı.

    YOLOv8 nesne tespiti + MOG2 hareket blob'larını birleştirir,
    her kaynak için ayrı tracker (ReID politikası farklı) çalıştırır
    ve prefix'li kalıcı ID üretir.
    """

    # COCO sınıf adları (yaygın endüstriyel sınıflar dahil)
    DEFAULT_NAMES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        80: "mog2_blob",  # sentetik sınıf
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        classes: Optional[List[int]] = None,
        yolo_tracker_cfg: Optional[Union[str, Path]] = None,
        mog_tracker_cfg: Optional[Union[str, Path]] = None,
        frame_rate: int = 30,
        mog2: Optional[MOG2Detector] = None,
        yolo_model: Any = None,
        use_ultralytics_botsort: bool = True,
        suppress_mog2_iou: float = 0.3,
    ) -> None:
        """
        Args:
            model_path: YOLO ağırlık dosyası (HF/ultralytics indirme).
            conf / iou: YOLO eşikleri.
            device: 'cpu' veya 'cuda'.
            classes: Filtrelenecek sınıf ID listesi (None = hepsi).
            yolo_tracker_cfg: ReID=true config yolu.
            mog_tracker_cfg: ReID=false config yolu.
            frame_rate: Video FPS (tracker buffer için).
            mog2: Harici MOG2Detector örneği (test enjeksiyonu).
            yolo_model: Harici YOLO modeli (test mock).
            use_ultralytics_botsort: True ise BOTSORT dene, yoksa IoU.
            suppress_mog2_iou: YOLO ile örtüşen MOG2 blob bastırma eşiği.
        """
        self.conf = conf
        self.iou = iou
        self.device = device
        self.classes = classes
        self.frame_rate = frame_rate
        self.suppress_mog2_iou = suppress_mog2_iou
        self._frame_idx = 0

        self.yolo_cfg_path = Path(yolo_tracker_cfg or DEFAULT_YOLO_TRACKER_CFG)
        self.mog_cfg_path = Path(mog_tracker_cfg or DEFAULT_MOG_TRACKER_CFG)
        self.yolo_args = load_tracker_args(self.yolo_cfg_path)
        self.mog_args = load_tracker_args(self.mog_cfg_path)

        # ReID politikasını zorunlu kıl (config sapmalarına karşı)
        self.yolo_args.with_reid = True
        self.mog_args.with_reid = False

        self.mog2 = mog2 or MOG2Detector()
        self._yolo = yolo_model
        self._model_path = model_path
        self._names: Dict[int, str] = dict(self.DEFAULT_NAMES)

        self._use_ultralytics_botsort = use_ultralytics_botsort

        self._yolo_tracker = self._build_tracker(
            self.yolo_args, use_ultralytics_botsort
        )
        self._mog_tracker = self._build_tracker(
            self.mog_args, use_ultralytics_botsort
        )

        # ReID bayraklarını dışarıdan doğrulanabilir tut
        self.yolo_with_reid: bool = bool(getattr(self.yolo_args, "with_reid", True))
        self.mog_with_reid: bool = bool(getattr(self.mog_args, "with_reid", False))

    def _build_tracker(self, args: SimpleNamespace, prefer_botsort: bool):
        """BoT-SORT veya SimpleIoUTracker oluştur."""
        if prefer_botsort:
            tracker = _try_create_botsort(args, frame_rate=self.frame_rate)
            if tracker is not None:
                return tracker
        return SimpleIoUTracker(
            match_thresh=float(getattr(args, "match_thresh", 0.3)),
            max_age=int(getattr(args, "track_buffer", 30)),
            with_reid=bool(getattr(args, "with_reid", False)),
        )

    @property
    def yolo(self):
        """YOLO modelini tembel yükle."""
        if self._yolo is None:
            from ultralytics import YOLO

            self._yolo = YOLO(self._model_path)
            # Sınıf isimleri
            if hasattr(self._yolo, "names") and self._yolo.names:
                self._names.update({int(k): str(v) for k, v in self._yolo.names.items()})
        return self._yolo

    def reset(self) -> None:
        """Tracker ve MOG2 durumunu sıfırla."""
        self._frame_idx = 0
        self.mog2.reset()
        # İlk kurulumdaki tercih korunur (test ortamında False kalmalı)
        self._yolo_tracker = self._build_tracker(self.yolo_args, self._use_ultralytics_botsort)
        self._mog_tracker  = self._build_tracker(self.mog_args,  self._use_ultralytics_botsort)

    def _class_name(self, class_id: int) -> str:
        return self._names.get(int(class_id), f"class_{class_id}")

    def _detect_yolo(self, frame: np.ndarray) -> np.ndarray:
        """
        YOLO inference → (N, 6) x1,y1,x2,y2,conf,cls

        Mock model: predict() yerine doğrudan dizi dönebilir.
        """
        # Test mock'u: callable veya .predict içeren nesne
        if hasattr(self._yolo, "predict") or self._yolo is None:
            model = self.yolo
            results = model.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                classes=self.classes,
                verbose=False,
            )
            if not results:
                return np.zeros((0, 6), dtype=np.float32)
            r0 = results[0]
            if r0.boxes is None or len(r0.boxes) == 0:
                return np.zeros((0, 6), dtype=np.float32)
            xyxy = r0.boxes.xyxy.cpu().numpy()
            conf = r0.boxes.conf.cpu().numpy().reshape(-1, 1)
            cls = r0.boxes.cls.cpu().numpy().reshape(-1, 1)
            return np.hstack([xyxy, conf, cls]).astype(np.float32)

        # Mock: model(frame) → ndarray
        out = self._yolo(frame)
        if out is None:
            return np.zeros((0, 6), dtype=np.float32)
        arr = np.asarray(out, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 6:
            pad = np.zeros((arr.shape[0], 6 - arr.shape[1]), dtype=np.float32)
            arr = np.hstack([arr, pad])
        return arr[:, :6]

    def _run_tracker(
        self, tracker: Any, detections: np.ndarray, frame: np.ndarray
    ) -> np.ndarray:
        """
        Tracker.update çağrısı — ultralytics ve SimpleIoU uyumlu.

        Ultralytics BOTSORT genelde (N, 6) xyxy+conf+cls bekler.
        """
        if detections is None or len(detections) == 0:
            detections = np.zeros((0, 6), dtype=np.float32)

        try:
            # Ultralytics BOTSORT: update(results, img) veya update(dets, img)
            out = tracker.update(detections, frame)
        except TypeError:
            try:
                out = tracker.update(detections, img=frame)
            except Exception:
                out = np.zeros((0, 7), dtype=np.float32)
        except Exception:
            # BoT-SORT bazen Results nesnesi ister; SimpleIoU'ya düş
            if not isinstance(tracker, SimpleIoUTracker):
                fallback = SimpleIoUTracker(
                    with_reid=bool(getattr(tracker, "with_reid", False))
                )
                out = fallback.update(detections, frame)
            else:
                out = np.zeros((0, 7), dtype=np.float32)

        if out is None:
            return np.zeros((0, 7), dtype=np.float32)
        arr = np.asarray(out, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, 7), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _parse_tracks(
        self,
        raw: np.ndarray,
        prefix: str,
        source: str,
        with_reid: bool,
        frame_idx: int,
        default_cls: int = 0,
    ) -> List[TrackedObject]:
        """Ham tracker çıktısını TrackedObject listesine çevir."""
        objects: List[TrackedObject] = []
        if raw is None or len(raw) == 0:
            return objects

        for row in raw:
            # Olası formatlar:
            # [x1,y1,x2,y2,track_id,conf,cls]  (SimpleIoU / bazı sürümler)
            # [x1,y1,x2,y2,conf,cls,track_id]  (ultralytics varyant)
            if len(row) < 5:
                continue
            x1, y1, x2, y2 = map(float, row[:4])
            track_num: int
            conf: float
            cls_id: int

            if len(row) >= 7:
                # Heuristik: ultralytics formatı [x1,y1,x2,y2, track_id, conf, cls]
                # veya bazı sürümlerde [x1,y1,x2,y2, conf, cls, track_id]
                # SimpleIoU:  [x1,y1,x2,y2, track_id(int≥1), conf(0-1), cls(int)]
                a4, a5, a6 = float(row[4]), float(row[5]), float(row[6])
                # a6'nın cls olup olmadığı en güvenilir ayrım:
                # cls 0-100 arası tam sayıdır; track_id ≥ 1 tam sayıdır.
                # [track_id, conf, cls]: a4 tam sayı ≥1, a5 ∈[0,1], a6 tam sayı ∈[0,100]
                # [conf, cls, track_id]: a4 ∈[0,1], a5 tam sayı ∈[0,100], a6 tam sayı ≥1
                a4_is_int = a4 == int(a4) and a4 >= 1
                a6_is_cls = a6 == int(a6) and 0 <= int(a6) <= 100
                a5_is_cls = a5 == int(a5) and 0 <= int(a5) <= 100

                if a4_is_int and a6_is_cls and not (0.0 <= a4 <= 1.0 and not a4_is_int):
                    # [track_id, conf, cls] — SimpleIoU / ultralytics varyant 1
                    track_num, conf, cls_id = int(a4), float(a5), int(a6)
                elif 0.0 <= a4 <= 1.0 and a5_is_cls:
                    # [conf, cls, track_id] — ultralytics varyant 2
                    conf, cls_id, track_num = float(a4), int(a5), int(a6)
                else:
                    # Fallback: SimpleIoU varsayılan format
                    track_num, conf, cls_id = int(a4), float(a5), int(a6)
            elif len(row) == 6:
                # x1..y2, track_id, conf
                track_num = int(row[4])
                conf = float(row[5])
                cls_id = default_cls
            else:
                track_num = int(row[4])
                conf = 0.5
                cls_id = default_cls

            bbox = (int(x1), int(y1), int(x2), int(y2))
            objects.append(
                TrackedObject(
                    track_id=f"{prefix}{track_num}",
                    bbox=bbox,
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=self._class_name(int(cls_id))
                    if source == "yolo"
                    else "mog2_blob",
                    source=source,
                    frame_idx=frame_idx,
                    with_reid=with_reid,
                )
            )
        return objects

    def process_frame(
        self,
        frame: np.ndarray,
        yolo_detections: Optional[np.ndarray] = None,
        run_mog2: bool = True,
    ) -> FrameTracks:
        """
        Tek kare işle: YOLO + MOG2 tespit → ayrı tracker → birleşik sonuç.

        Args:
            frame: BGR görüntü.
            yolo_detections: Verilirse YOLO inference atlanır (test için).
            run_mog2: False ise sadece YOLO hattı.

        Returns:
            FrameTracks — prefix'li tüm track'ler.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Boş kare verildi.")

        frame_idx = self._frame_idx
        self._frame_idx += 1

        # --- YOLO hattı (ReID açık) ---
        if yolo_detections is not None:
            yolo_dets = np.asarray(yolo_detections, dtype=np.float32)
            if yolo_dets.size == 0:
                yolo_dets = np.zeros((0, 6), dtype=np.float32)
            elif yolo_dets.ndim == 1:
                yolo_dets = yolo_dets.reshape(1, -1)
        else:
            yolo_dets = self._detect_yolo(frame)

        yolo_raw = self._run_tracker(self._yolo_tracker, yolo_dets, frame)
        yolo_tracks = self._parse_tracks(
            yolo_raw,
            prefix=YOLO_PREFIX,
            source="yolo",
            with_reid=self.yolo_with_reid,
            frame_idx=frame_idx,
        )

        # --- MOG2 hattı (ReID kapalı) ---
        mog_tracks: List[TrackedObject] = []
        mog_det_count = 0
        if run_mog2:
            yolo_boxes = [t.bbox for t in yolo_tracks]
            # Ham YOLO kutularını da bastırma için ekle
            if len(yolo_dets) > 0:
                for row in yolo_dets:
                    yolo_boxes.append(
                        (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
                    )

            blobs: List[BlobDetection] = self.mog2.detect(
                frame,
                exclude_boxes=yolo_boxes if yolo_boxes else None,
                iou_suppress=self.suppress_mog2_iou,
            )
            mog_det_count = len(blobs)
            mog_arr = self.mog2.to_xyxy_conf(blobs)
            # cls sütunu ekle (sentetik sınıf 80)
            if len(mog_arr) > 0:
                cls_col = np.full((len(mog_arr), 1), 80.0, dtype=np.float32)
                mog_dets = np.hstack([mog_arr, cls_col])
            else:
                mog_dets = np.zeros((0, 6), dtype=np.float32)

            # Güvenlik: MOG tracker ReID kapalı olmalı
            if hasattr(self._mog_tracker, "with_reid"):
                self._mog_tracker.with_reid = False
            if hasattr(self.mog_args, "with_reid"):
                assert self.mog_args.with_reid is False, "MOG2 ReID kapalı olmalı"

            mog_raw = self._run_tracker(self._mog_tracker, mog_dets, frame)
            mog_tracks = self._parse_tracks(
                mog_raw,
                prefix=MOG_PREFIX,
                source="mog2",
                with_reid=False,  # her zaman False
                frame_idx=frame_idx,
                default_cls=80,
            )

        all_tracks = yolo_tracks + mog_tracks
        return FrameTracks(
            frame_idx=frame_idx,
            tracks=all_tracks,
            yolo_detections=int(len(yolo_dets)),
            mog2_detections=mog_det_count,
        )

    def process_detections(
        self,
        yolo_detections: np.ndarray,
        mog2_detections: np.ndarray,
        frame: Optional[np.ndarray] = None,
        frame_idx: Optional[int] = None,
    ) -> FrameTracks:
        """
        Ham tespit dizileriyle tracker güncelle (unit test için ideal).

        Args:
            yolo_detections: (N, 6) xyxy conf cls
            mog2_detections: (M, 5) xyxy conf  veya (M, 6)
            frame: Opsiyonel görüntü (BoT-SORT GMC için).
            frame_idx: Opsiyonel kare indeksi.
        """
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        if frame_idx is not None:
            self._frame_idx = frame_idx
        idx = self._frame_idx
        self._frame_idx += 1

        yolo_dets = np.asarray(yolo_detections, dtype=np.float32)
        if yolo_dets.size == 0:
            yolo_dets = np.zeros((0, 6), dtype=np.float32)
        elif yolo_dets.ndim == 1:
            yolo_dets = yolo_dets.reshape(1, -1)

        mog_dets = np.asarray(mog2_detections, dtype=np.float32)
        if mog_dets.size == 0:
            mog_dets = np.zeros((0, 6), dtype=np.float32)
        elif mog_dets.ndim == 1:
            mog_dets = mog_dets.reshape(1, -1)
        if mog_dets.shape[1] == 5:
            cls_col = np.full((len(mog_dets), 1), 80.0, dtype=np.float32)
            mog_dets = np.hstack([mog_dets, cls_col])

        yolo_raw = self._run_tracker(self._yolo_tracker, yolo_dets, frame)
        mog_raw = self._run_tracker(self._mog_tracker, mog_dets, frame)

        yolo_tracks = self._parse_tracks(
            yolo_raw, YOLO_PREFIX, "yolo", self.yolo_with_reid, idx
        )
        mog_tracks = self._parse_tracks(
            mog_raw, MOG_PREFIX, "mog2", False, idx, default_cls=80
        )

        return FrameTracks(
            frame_idx=idx,
            tracks=yolo_tracks + mog_tracks,
            yolo_detections=len(yolo_dets),
            mog2_detections=len(mog_dets),
        )

    def get_reid_policy(self) -> Dict[str, bool]:
        """Kaynak bazlı ReID politikasını döndür (test doğrulaması için)."""
        return {
            "yolo": self.yolo_with_reid,
            "mog2": self.mog_with_reid,
        }
