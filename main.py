from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def build_output_path(input_path: Path) -> Path:
    # 입력 파일명이 input.jpg 이면 결과 파일명을 input_cartoon.jpg 로 만든다.
    return input_path.with_name(f"{input_path.stem}_cartoon{input_path.suffix}")


def posterize(image: np.ndarray, levels: int = 8) -> np.ndarray:
    # 색 단계를 줄이면 사진 같은 연속 톤이 줄어들고,
    # 만화처럼 몇 개의 평평한 색 면으로 보이기 쉬워진다.
    step = max(1, 256 // levels)
    image_i16 = image.astype(np.int16)
    reduced = (image_i16 // step) * step + step // 2
    return np.clip(reduced, 0, 255).astype(np.uint8)


def boost_color(image: np.ndarray) -> np.ndarray:
    # 채도와 밝기를 조금 올려서 만화 같은 선명한 느낌을 더한다.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def make_edge_overlay(image: np.ndarray) -> np.ndarray:
    # 1. 윤곽선 검출은 색보다 밝기 변화가 중요하므로 grayscale 로 변환한다.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. edge 전에 작은 노이즈를 줄이면 배경의 잔선이 덜 생긴다.
    gray = cv2.medianBlur(gray, 5)

    # 3. Canny 로 윤곽선을 검출한다.
    # 낮은 값과 높은 값을 함께 써서 강한 경계와 약한 경계를 구분한다.
    edges = cv2.Canny(gray, 80, 150)

    # 4. 선이 너무 가늘면 잘 안 보일 수 있어서 살짝 두껍게 만든다.
    kernel = np.ones((2, 2), dtype=np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 5. bitwise_and 에 바로 쓰기 위해 흰 배경 / 검은 선 형태로 뒤집고,
    # 3채널 이미지로 바꿔 컬러 이미지와 같은 형태로 맞춘다.
    edges = cv2.bitwise_not(edges)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def smooth_colors(image: np.ndarray) -> np.ndarray:
    # bilateral filter 를 약하게 여러 번 적용하면
    # 경계는 비교적 유지하면서 색 면만 부드럽게 만들 수 있다.
    color = image.copy()
    for _ in range(3):
        color = cv2.bilateralFilter(color, 9, 50, 50)
    return color


def cartoonize(image: np.ndarray) -> np.ndarray:
    # 1. 먼저 색 면을 부드럽게 정리한다.
    color = smooth_colors(image)

    # 2. 색 단계를 줄여서 사진 느낌을 줄이고 만화 느낌을 강하게 만든다.
    color = posterize(color, levels=8)

    # 3. 채도와 밝기를 조금 올려 색을 더 또렷하게 만든다.
    color = boost_color(color)

    # 4. 윤곽선을 따로 검출해서 검은 선처럼 합성한다.
    edge_overlay = make_edge_overlay(image)

    # 5. 색 이미지와 윤곽선을 합쳐 최종 카툰 이미지를 만든다.
    return cv2.bitwise_and(color, edge_overlay)


def main() -> int:
    # 사용법은 python main.py input.jpg 하나만 받는다.
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_image>", file=sys.stderr)
        return 1

    input_path = Path(sys.argv[1])
    output_path = build_output_path(input_path)

    # 이미지를 읽고 실패하면 종료한다.
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Could not read image: {input_path}", file=sys.stderr)
        return 1

    # 카툰 변환 후 자동으로 input_cartoon.jpg 형태로 저장한다.
    cartoon = cartoonize(image)
    if not cv2.imwrite(str(output_path), cartoon):
        print(f"Could not write output image: {output_path}", file=sys.stderr)
        return 1

    print(f"Saved cartoon image to: {output_path}")

    # 저장 후 바로 결과를 화면에 띄운다.
    try:
        cv2.imshow("Cartoon Rendering", cartoon)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print(
            "OpenCV GUI support is not available in this environment. "
            f"Open the saved file manually: {output_path}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
