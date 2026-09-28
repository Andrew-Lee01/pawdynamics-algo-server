"""
imu_terrain_calib.py — IMU 지형 보정 계수 생성
================================================
설계자료 5-2절 + 2학기 IMU 활용 기반.

목적:
    오르막/내리막에서 GRF 가 앞뒤 발로 쏠리는 것을
    IMU pitch 값으로 보정하는 계수를 실측으로 검증한다.

이론:
    GRF_corrected = GRF_raw × cos(pitch_rad)

    평지(0°):  cos(0)  = 1.000 → 보정 없음
    오르막 15°: cos(15°) = 0.966 → 6% 감소
    내리막 15°: cos(-15°) = 0.966 → 동일

실측 검증:
    1. 평지에서 기준 GRF 측정
    2. 오르막/내리막에서 동일 하중으로 측정
    3. 이론값 vs 실측값 비교 → 보정 계수 저장

저장:
    calib/terrain.json
    {"flat": 1.0, "uphill_15": 0.967, "downhill_15": 0.964, ...}

[2026-09-28] get_realtime_correction() 이 calib/terrain.json 의 실측값을 실제로
읽어서 선형보간하도록 수정 (예전엔 파일을 저장만 하고 실시간 파이프라인은
이론값(cos(pitch))만 썼음 — 측정해도 반영이 안 되던 문제).
"""

import asyncio
import json
import math
import numpy as np
import logging
from pathlib import Path

from ble_receiver import DummyPawReceiver, PawReceiver
from frame_sync import FrameSynchronizer, SyncedFrame, NUM_PAWS
from node_calibration import load_calibration, apply_calibration
from cross_sensor_calib import load_gains

LF = 0   # 실시간 파이프라인과 동일하게 IMU pitch 는 LF 기준

log = logging.getLogger("TERRAIN_CALIB")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ── 설정 ──────────────────────────────────────
TERRAIN_CONDITIONS = [
    ("flat",         0.0,   "평지"),
    ("uphill_10",   10.0,   "오르막 10°"),
    ("uphill_15",   15.0,   "오르막 15°"),
    ("downhill_10",-10.0,   "내리막 10°"),
    ("downhill_15",-15.0,   "내리막 15°"),
]
CALIB_FRAMES = 100
SAVE_PATH    = Path("calib/terrain.json")
USE_DUMMY    = False  # 실제 보드 사용 시 False


# ── 지형 보정 수집기 ───────────────────────────
class TerrainCalibrator:

    def __init__(self):
        self._frames = []
        self._collecting = False
        self._node_coeffs = [load_calibration(pos) for pos in range(NUM_PAWS)]
        self._gains = load_gains()

    def on_synced(self, synced: SyncedFrame):
        if not self._collecting:
            return
        total_sum = 0.0
        for pos in range(NUM_PAWS):
            mat = synced.get_matrix(pos)
            corrected = apply_calibration(mat, self._node_coeffs[pos])
            raw_sum = corrected.sum() * self._gains.get(pos, 1.0)
            total_sum += raw_sum

        # IMU pitch 는 실시간 파이프라인(gait_realtime_v2)과 동일하게 LF 기준만 사용
        imu_pitch = synced.get_imu(LF).pitch_deg

        self._frames.append({
            "grf_sum":   total_sum,
            "imu_pitch": imu_pitch,
        })

    def start_collection(self):
        self._collecting = True
        self._frames.clear()

    def stop_collection(self):
        self._collecting = False

    def collected_count(self) -> int:
        return len(self._frames)

    def get_stats(self) -> dict:
        if not self._frames:
            return {"grf_mean": 0.0, "pitch_mean": 0.0}
        grf_list   = [f["grf_sum"]   for f in self._frames]
        pitch_list = [f["imu_pitch"] for f in self._frames]
        return {
            "grf_mean":   float(np.mean(grf_list)),
            "grf_std":    float(np.std(grf_list)),
            "pitch_mean": float(np.mean(pitch_list)),
            "pitch_std":  float(np.std(pitch_list)),
        }


# ── 보정 계수 계산 ─────────────────────────────
def compute_terrain_coeffs(results: dict) -> dict:
    """
    평지 GRF 를 기준으로 각 지형의 보정 계수 계산.

    이론값: cos(pitch)
    실측값: grf_terrain / grf_flat

    최종 계수 = (이론값 + 실측값) / 2  [두 값 평균]
    """
    flat_grf = results.get("flat", {}).get("grf_mean", 1.0)
    if flat_grf < 1e-6:
        log.error("평지 GRF 기준값이 없음")
        return {}

    coeffs = {}
    for key, pitch_deg, label in TERRAIN_CONDITIONS:
        stats = results.get(key, {})
        grf_mean = stats.get("grf_mean", flat_grf)
        pitch_measured = stats.get("pitch_mean", pitch_deg)

        # 이론 보정 계수
        theory = math.cos(math.radians(pitch_deg))
        # 실측 보정 계수
        measured = grf_mean / flat_grf if flat_grf > 0 else 1.0

        # 평균
        final = (theory + measured) / 2.0
        final = float(np.clip(final, 0.5, 1.0))

        coeffs[key] = {
            "pitch_deg":      pitch_deg,
            "label":          label,
            "coeff_theory":   round(theory, 4),
            "coeff_measured": round(measured, 4),
            "coeff_final":    round(final, 4),
            "pitch_measured": round(pitch_measured, 2),
        }
        log.info(f"{label}: 이론={theory:.4f} 실측={measured:.4f} "
                 f"최종={final:.4f}")

    return coeffs


def save_terrain(coeffs: dict):
    SAVE_PATH.parent.mkdir(exist_ok=True)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(coeffs, f, ensure_ascii=False, indent=2)
    log.info(f"지형 보정 저장: {SAVE_PATH}")
    _invalidate_terrain_cache()   # 같은 프로세스에서 바로 이어서 써도 최신값 반영되게


def load_terrain() -> dict:
    """저장된 지형 보정 불러오기"""
    if not SAVE_PATH.exists():
        log.warning(f"지형 보정 파일 없음: {SAVE_PATH} → 기본값 사용")
        return {key: {"coeff_final": math.cos(math.radians(pitch))}
                for key, pitch, _ in TERRAIN_CONDITIONS}
    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# get_realtime_correction() 이 파일을 매 프레임(50Hz)마다 다시 읽지 않도록 캐싱.
# 캐싱된 뒤에 새로 보정(run_terrain_calibration)을 돌리면 _invalidate_terrain_cache()
# 로 비워줘야 같은 프로세스 안에서도 최신값을 씀(별도 프로세스로 켜면 신경 안 써도 됨).
_terrain_points_cache: list | None = None


def _invalidate_terrain_cache():
    global _terrain_points_cache
    _terrain_points_cache = None


def _load_terrain_points() -> list | None:
    """calib/terrain.json → (pitch_deg, coeff_final) 오름차순 정렬 리스트.
    파일이 없거나 비어있으면 None (이론값으로 대체하라는 신호)."""
    global _terrain_points_cache
    if _terrain_points_cache is not None:
        return _terrain_points_cache
    if not SAVE_PATH.exists():
        return None
    try:
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    points = [
        (pitch_deg, entry["coeff_final"])
        for key, pitch_deg, _label in TERRAIN_CONDITIONS
        if (entry := data.get(key)) and "coeff_final" in entry
    ]
    if not points:
        return None
    points.sort(key=lambda p: p[0])
    _terrain_points_cache = points
    return points


def get_realtime_correction(pitch_deg: float) -> float:
    """
    실시간 pitch 값으로 GRF 보정 계수 반환.

    calib/terrain.json 에 실측 보정값(run_terrain_calibration() 결과)이 있으면
    그 점들(평지/±10°/±15°)을 선형보간해서 쓴다. 측정한 각도 사이 값은 보간,
    측정 범위(±15°) 밖은 양 끝 값으로 고정(외삽 안 함 — 측정 안 해본 각도라 신뢰 못 함).
    아직 측정 전(파일 없음)이면 이론값 cos(pitch) 로 대체한다.

    Args:
        pitch_deg: IMU 에서 읽은 pitch (°)

    Returns:
        보정 계수 (0.5 ~ 1.0)
    """
    points = _load_terrain_points()
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        factor = float(np.interp(pitch_deg, xs, ys))   # 범위 밖은 자동으로 끝값 고정
    else:
        factor = math.cos(math.radians(pitch_deg))
    return float(np.clip(factor, 0.5, 1.0))


# ── 메인 ──────────────────────────────────────
async def run_terrain_calibration():
    calibrator = TerrainCalibrator()
    sync = FrameSynchronizer(on_synced=calibrator.on_synced)

    if USE_DUMMY:
        receiver = DummyPawReceiver(on_frame=sync.on_frame)
        recv_task = asyncio.create_task(receiver.run(hz=50))
    else:
        receiver = PawReceiver(on_frame=sync.on_frame)
        recv_task = asyncio.create_task(receiver.run())
        await asyncio.sleep(15.0)

    results = {}

    for key, pitch_deg, label in TERRAIN_CONDITIONS:
        input(f"\n[지형보정] {label} ({pitch_deg}°) 에 위치 후 Enter...")

        log.info(f"{label} 측정 시작...")
        calibrator.start_collection()

        while calibrator.collected_count() < CALIB_FRAMES:
            await asyncio.sleep(0.1)

        calibrator.stop_collection()
        stats = calibrator.get_stats()
        results[key] = stats
        log.info(f"{label}: GRF={stats['grf_mean']:.1f} "
                 f"pitch={stats['pitch_mean']:.1f}°")

    # 보정 계수 계산 및 저장
    coeffs = compute_terrain_coeffs(results)
    save_terrain(coeffs)

    recv_task.cancel()
    try:
        await recv_task
    except asyncio.CancelledError:
        pass

    log.info("지형 보정 완료!")


if __name__ == "__main__":
    print("=== PawDynamics IMU 지형 보정 (LF 기준, 2발 모드) ===")
    print("⚠ node_calibration.py 와 cross_sensor_calib.py 를 먼저 실행하세요")
    print(f"측정 조건: {[c[2] for c in TERRAIN_CONDITIONS]}")
    if USE_DUMMY:
        print("※ 더미 모드")
    asyncio.run(run_terrain_calibration())
