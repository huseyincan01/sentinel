"""
MOG2 arkaplan çıkarımı ve blob tespiti.

OpenCV createBackgroundSubtractorMOG2 kullanarak hareketli bölgeleri
tespit eder, gürültüyü filtreler ve YOLO ile birleştirilebilecek
bounding box listesi üretir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class BlobDetection:
    """Tek bir MOG2 blob tespiti."""

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    area: float
    confidence: float = 0.5  # MOG2 için sentetik güven skoru
    source: str = "mog2"


class MOG2Detector:
    """
    MOG2 tabanlı hareket/blob dedektörü.

    YOLO'nun kaçırabileceği şekilsiz hareketleri (duman, dökülme,
    ani siluet değişimi vb.) yakalamak için kullanılır.
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        min_area: int = 500,
        max_area_ratio: float = 0.4,
        morph_kernel_size: int = 5,
        confidence: float = 0.5,
    ) -> None:
        """
        Args:
            history: Arkaplan modelinde tutulan kare sayısı.
            var_threshold: Varyans eşiği (düşük = daha hassas).
            detect_shadows: Gölge işaretlemesi (gri piksel).
            min_area: Minimum blob alanı (piksel^2) — gürültü filtresi.
            max_area_ratio: Kare alanına göre maksimum blob oranı.
            morph_kernel_size: Morfolojik açma/kapama kernel boyutu.
            confidence: Blob'lara atanan sabit güven skoru.
        """
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.confidence = confidence
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        # OpenCV MOG2 arkaplan çıkarıcı
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self._frame_shape: Optional[Tuple[int, int]] = None  # (h, w)
        self.last_cleaned_mask: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Arkaplan modelini sıfırla (yeni video başlangıcı)."""
        history = self.bg_subtractor.getHistory()
        var_threshold = self.bg_subtractor.getVarThreshold()
        detect_shadows = self.bg_subtractor.getDetectShadows()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self._frame_shape = None
        self.last_cleaned_mask = None

    def apply(self, frame: np.ndarray, learning_rate: float = -1) -> np.ndarray:
        """
        Tek kare üzerinde arkaplan çıkarımı uygula.

        Returns:
            Binary (veya gölgeli) foreground maskesi.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Boş kare verildi.")
        self._frame_shape = frame.shape[:2]
        # learning_rate=-1: OpenCV otomatik öğrenme hızı kullanır
        fg_mask = self.bg_subtractor.apply(frame, learningRate=learning_rate)
        return fg_mask

    def _clean_mask(self, fg_mask: np.ndarray) -> np.ndarray:
        """Gölge ve gürültüyü temizle; morfolojik işlem uygula."""
        # Gölgeler 127 değeriyle işaretlenir; yalnızca kesin FG (255) kalsın
        _, binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        # Açma: küçük gürültü; kapama: delikleri doldur
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self._kernel)
        return cleaned

    def detect(
        self,
        frame: np.ndarray,
        learning_rate: float = -1,
        exclude_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
        iou_suppress: float = 0.3,
    ) -> List[BlobDetection]:
        """
        Kareden hareketli blob'ları tespit et.

        Args:
            frame: BGR görüntü.
            learning_rate: MOG2 öğrenme hızı.
            exclude_boxes: YOLO kutuları — yüksek IoU'lu blob'lar elenir
                           (çift sayımı azaltmak için).
            iou_suppress: exclude_boxes ile IoU eşiği; üstü elenir.

        Returns:
            Filtrelenmiş BlobDetection listesi.
        """
        fg_mask = self.apply(frame, learning_rate=learning_rate)
        cleaned = self._clean_mask(fg_mask)
        self.last_cleaned_mask = cleaned
        h, w = cleaned.shape[:2]
        max_area = (h * w) * self.max_area_ratio

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[BlobDetection] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            bbox = (x, y, x + bw, y + bh)

            # YOLO ile örtüşen blob'ları bastır
            if exclude_boxes and self._overlaps_any(bbox, exclude_boxes, iou_suppress):
                continue

            detections.append(
                BlobDetection(
                    bbox=bbox,
                    area=area,
                    confidence=self.confidence,
                    source="mog2",
                )
            )
        return detections

    @staticmethod
    def _iou(
        a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
    ) -> float:
        """İki bbox arasında IoU hesapla."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    def _overlaps_any(
        self,
        bbox: Tuple[int, int, int, int],
        boxes: Sequence[Tuple[int, int, int, int]],
        threshold: float,
    ) -> bool:
        return any(self._iou(bbox, other) >= threshold for other in boxes)

    def to_xyxy_conf(
        self, detections: Sequence[BlobDetection]
    ) -> np.ndarray:
        """
        Blob listesini (N, 5) array'e çevir: x1,y1,x2,y2,conf.

        BoT-SORT / ultralytics beslemesi için uygun format.
        """
        if not detections:
            return np.zeros((0, 5), dtype=np.float32)
        rows = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            rows.append([x1, y1, x2, y2, d.confidence])
        return np.asarray(rows, dtype=np.float32)
