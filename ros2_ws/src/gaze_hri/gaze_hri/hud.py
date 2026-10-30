#!/usr/bin/env python3
"""
탑다운 화면 HUD 그리기.

`topdown_click_node` 의 화면 표시를 여기로 분리했습니다. 노드는 "무엇을 보여줄지"만
정하고, "어떻게 그릴지"는 전부 이 파일에 있습니다.

한글 표시
---------
OpenCV의 `putText` 는 한글을 못 그립니다(물음표로 나옵니다). 그래서 Pillow 가
있고 한글 폰트를 찾으면 그걸로 그리고, 없으면 영문 문구로 자동 폴백합니다.
둘 다 같은 화면 구성을 유지하므로 폰트가 없어도 기능은 그대로입니다.

  Pillow 없음 / 폰트 없음  ->  영문 라벨로 표시 (동작에는 지장 없음)
"""

import os

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL = True
except ImportError:                                    # pragma: no cover
    _PIL = False

# 한글 폰트 후보. 배포판마다 다르므로 순서대로 찾는다.
_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]

# --- 색 (BGR) ---
PANEL = (30, 27, 24)          # 패널 바탕
FG = (242, 242, 242)          # 기본 글자
MUTED = (155, 150, 145)       # 보조 글자
IDLE = (255, 190, 80)         # 하늘색  — 고를 차례
PICK = (60, 150, 255)         # 주황    — 집을 것
PLACE = (110, 225, 130)       # 초록    — 놓을 곳
BUSY = (90, 210, 255)         # 노랑    — 동작 중
WARN = (85, 85, 255)          # 빨강    — 경고
OBJ = (190, 185, 100)         # 청록    — 검출된 물체


def _find_font():
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


class Hud:
    """화면 그리기 도구. 폰트를 한 번만 로드해서 재사용합니다."""

    def __init__(self):
        self.font_path = _find_font() if _PIL else None
        self.korean = self.font_path is not None
        self._cache = {}

    # ------------------------------------------------------------------
    def _font(self, size, bold=False):
        key = (size, bold)
        if key not in self._cache:
            idx = 1 if bold and self.font_path.endswith(".ttc") else 0
            try:
                self._cache[key] = ImageFont.truetype(self.font_path, size, index=idx)
            except Exception:                           # pragma: no cover
                self._cache[key] = ImageFont.truetype(self.font_path, size)
        return self._cache[key]

    def label(self, ko, en):
        """한글을 그릴 수 있으면 한글, 아니면 영문을 돌려준다."""
        return ko if self.korean else en

    def text_size(self, s, size=14, bold=False):
        if not self.korean:
            scale = size / 28.0
            (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            return w, h
        f = self._font(size, bold)
        box = f.getbbox(s)
        return box[2] - box[0], box[3] - box[1]

    def text(self, img, xy, s, color=FG, size=14, bold=False):
        """텍스트를 그린다. 반환값은 없고 img 를 제자리에서 수정한다."""
        if not s:
            return img
        if not self.korean:
            scale = size / 28.0
            cv2.putText(img, s, (int(xy[0]), int(xy[1] + size * 0.8)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            return img
        # PIL 은 RGB 이므로 색 순서를 뒤집어 넘긴다
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text((int(xy[0]), int(xy[1])), s,
                                 font=self._font(size, bold),
                                 fill=(color[2], color[1], color[0]))
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return img

    def texts(self, img, items):
        """여러 줄을 한 번에 그린다 (PIL 변환을 한 번만 하려고).

        items: [(xy, s, color, size, bold), ...]
        """
        items = [it for it in items if it[1]]
        if not items:
            return img
        if not self.korean:
            for (xy, s, color, size, bold) in items:
                self.text(img, xy, s, color, size, bold)
            return img
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for (xy, s, color, size, bold) in items:
            draw.text((int(xy[0]), int(xy[1])), s, font=self._font(size, bold),
                      fill=(color[2], color[1], color[0]))
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return img

    # ------------------------------------------------------------------
    @staticmethod
    def panel(img, x, y, w, h, alpha=0.72, color=PANEL):
        """반투명 패널. 글자가 영상 위에서도 읽히게 해준다."""
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(img.shape[1], int(x + w)), min(img.shape[0], int(y + h))
        if x1 <= x0 or y1 <= y0:
            return img
        roi = img[y0:y1, x0:x1]
        overlay = np.full_like(roi, color, dtype=np.uint8)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
        return img

    @staticmethod
    def accent_bar(img, x, y, w, h, color):
        # cv2.rectangle 은 끝점을 포함하므로 -1 해야 정확히 w x h 가 된다.
        # (안 하면 헤더 액센트가 아래 영상 영역을 1픽셀 침범한다)
        cv2.rectangle(img, (int(x), int(y)),
                      (int(x + w) - 1, int(y + h) - 1), color, -1)

    # ------------------------------------------------------------------
    def chip(self, img, xy, s, color=FG, size=13, bold=False, anchor="lt",
             bg=(24, 22, 20), alpha=0.78, pad=(7, 4)):
        """글자 뒤에 반투명 칩을 깔아준다.

        카메라 영상 위에 그냥 글자를 얹으면 배경 색에 따라 읽히지 않는다.
        anchor: lt(좌상) / ct(가운데 위) / cb(가운데 아래)
        """
        tw, th = self.text_size(s, size, bold)
        bw, bh = tw + pad[0] * 2, th + pad[1] * 2 + 4
        x, y = float(xy[0]), float(xy[1])
        if anchor == "ct":
            x, y = x - bw / 2, y - bh
        elif anchor == "cb":
            x = x - bw / 2
        x = max(0, min(img.shape[1] - bw, x))
        y = max(0, min(img.shape[0] - bh, y))
        self.panel(img, x, y, bw, bh, alpha=alpha, color=bg)
        self.text(img, (x + pad[0], y + pad[1]), s, color, size, bold)
        return x, y, bw, bh

    def badge(self, img, px, s, color, radius=11):
        """작은 원형 배지(컵 번호 등). 영상 위에서도 또렷하게."""
        u, v = int(px[0]), int(px[1])
        cv2.circle(img, (u, v), radius, (22, 20, 18), -1, cv2.LINE_AA)
        cv2.circle(img, (u, v), radius, color, 2, cv2.LINE_AA)
        tw, th = self.text_size(s, 13, True)
        self.text(img, (u - tw / 2, v - th / 2 - 2), s, color, 13, bold=True)

    @staticmethod
    def crosshair(img, px, color, size=13, thickness=1):
        u, v = int(px[0]), int(px[1])
        cv2.line(img, (u - size, v), (u - 4, v), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u + 4, v), (u + size, v), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u, v - size), (u, v - 4), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u, v + 4), (u, v + size), color, thickness, cv2.LINE_AA)

    @staticmethod
    def progress_ring(img, px, frac, color, radius=20):
        """응시 진행률을 원호로. 다 차면 선택된다는 뜻."""
        u, v = int(px[0]), int(px[1])
        cv2.circle(img, (u, v), radius, (70, 70, 70), 2, cv2.LINE_AA)
        if frac > 0:
            cv2.ellipse(img, (u, v), (radius, radius), -90, 0,
                        int(360 * min(1.0, frac)), color, 3, cv2.LINE_AA)

    @staticmethod
    def marker(img, px, color, radius=22, thickness=2, dashed=False):
        u, v = int(px[0]), int(px[1])
        if not dashed:
            cv2.circle(img, (u, v), radius, color, thickness, cv2.LINE_AA)
            return
        for a in range(0, 360, 30):
            cv2.ellipse(img, (u, v), (radius, radius), 0, a, a + 16,
                        color, thickness, cv2.LINE_AA)
