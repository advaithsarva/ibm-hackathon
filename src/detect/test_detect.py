"""Checks for the detection stack.

Covers the parts that run without weights: POS rPPG, thermal blob thresholding, RGB-T
merging, log-odds fusion and the GPU guard. YOLO and YAMNet are exercised only for
their failure paths, since their weights are not in the repo.

    python -m src.detect.test_detect      (or: pytest src/detect/test_detect.py)
"""
import numpy as np

from src.detect import rppg
from src.detect.fusion import (estimate_count, fuse_cell, fuse_log_odds, iou,
                               late_nms_merge, sensor_key)
from src.detect.loader import active, gpu_model
from src.detect.thermal import adaptive_threshold, detect_hotspots
from src.hazard.zones import load_global

RATIOS = load_global()["fusion"]["likelihood_ratios"]


def test_pos_recovers_a_known_heart_rate():
    """The whole point of POS: a pulse buried in a noisy colour trace comes back out."""
    for truth in (55.0, 72.0, 110.0):
        trace = rppg.synthetic_trace(bpm=truth, fps=30.0, seconds=12.0, noise=0.005)
        result = rppg.estimate(trace, fps=30.0)
        assert result["locked"], f"{truth} bpm: no lock at snr={result['snr_db']}"
        # 12 s at 30 fps resolves to about 5 bpm, so this is bin-limited, not sloppy.
        assert abs(result["bpm"] - truth) <= 4.0, \
            f"recovered {result['bpm']} bpm, expected {truth}"


def test_pos_refuses_when_the_pulse_sits_at_the_noise_floor():
    """Silence beats a confident wrong number when a rescue team reads the screen."""
    buried = rppg.synthetic_trace(bpm=72.0, fps=30.0, seconds=12.0, noise=0.02)
    assert not rppg.estimate(buried, fps=30.0)["locked"]


def test_pos_refuses_rather_than_inventing_a_pulse():
    """Pure noise must not produce a confident BPM. A rescue team acts on these."""
    rng = np.random.default_rng(1)
    noise = 0.5 + rng.normal(0, 0.05, (360, 3))
    result = rppg.estimate(noise, fps=30.0)
    assert not result["locked"], f"locked onto noise at {result['bpm']} bpm"
    assert result["bpm"] is None and result["conf"] == 0.0


def test_pos_rejects_bad_input():
    for bad in (lambda: rppg.pos_pulse(np.zeros((10, 2)), 30.0),      # wrong shape
                lambda: rppg.pos_pulse(np.zeros((300, 3)), 0),        # fps <= 0
                lambda: rppg.pos_pulse(np.zeros((5, 3)), 30.0)):      # too few frames
        try:
            bad()
            raise AssertionError("expected ValueError on invalid rPPG input")
        except ValueError:
            pass


def test_bandpass_keeps_the_physiological_band():
    fps, n = 30.0, 600
    t = np.arange(n) / fps
    in_band = np.sin(2 * np.pi * 1.2 * t)      # 72 bpm
    out_of_band = np.sin(2 * np.pi * 8.0 * t)  # 480 bpm, impossible
    filtered = rppg.bandpass(in_band + out_of_band, fps)
    assert np.corrcoef(filtered, in_band)[0, 1] > 0.95, "band content was attenuated"
    bpm, _ = rppg.heart_rate(filtered, fps)
    assert abs(bpm - 72.0) < 4.0


def test_thermal_threshold_adapts_to_the_scene():
    cold = np.full((40, 40), 5.0)
    hot = np.full((40, 40), 33.0)
    # Never drops below the coldest plausible human reading, even in a freezing scene.
    assert adaptive_threshold(cold) >= 26.0
    # In a uniformly warm scene the percentile rises above that floor.
    assert adaptive_threshold(hot) >= 26.0

    try:
        adaptive_threshold(np.zeros((3, 3, 3)))
        raise AssertionError("expected ValueError on a non-2D frame")
    except ValueError:
        pass


def test_thermal_finds_a_warm_body_and_flags_a_fire():
    frame = np.full((60, 60), 18.0)          # cool rubble
    frame[10:20, 10:20] = 33.5               # person-shaped warm patch
    frame[40:50, 40:50] = 200.0              # burning car

    hotspots = detect_hotspots(frame)
    kinds = {d["classification"] for d in hotspots}
    assert "human_range" in kinds, "missed the body-temperature blob"
    assert "too_hot_for_a_person" in kinds, "fire should be reported, not dropped"

    human = next(d for d in hotspots if d["classification"] == "human_range")
    fire = next(d for d in hotspots if d["classification"] == "too_hot_for_a_person")
    assert human["conf"] > fire["conf"], "a fire must not outrank a survivor"
    assert human["pixels"] == 100

    # Speckle below the minimum blob size is noise, not a person.
    speckle = np.full((60, 60), 18.0)
    speckle[5, 5] = 35.0
    assert detect_hotspots(speckle) == []


def test_rgbt_merge_counts_one_person_once():
    rgb = [{"type": "rgb_person", "conf": 0.66, "bbox": [100, 100, 150, 200]}]
    thermal = [{"type": "thermal_hotspot", "conf": 0.71, "temp_c": 34.2,
                "bbox": [105, 98, 152, 205]}]
    merged = late_nms_merge(rgb, thermal)
    assert len(merged) == 1 and merged[0]["type"] == "rgb_thermal"
    assert merged[0]["rgb_conf"] == 0.66 and merged[0]["thermal_conf"] == 0.71

    # Boxes far apart are two different people and must both survive.
    apart = late_nms_merge(rgb, [{"type": "thermal_hotspot", "conf": 0.6, "temp_c": 33.0,
                                  "bbox": [400, 400, 440, 480]}])
    assert len(apart) == 2

    # A thermal hit with no RGB partner still passes through: smoke blinds cameras.
    assert len(late_nms_merge([], thermal)) == 1
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [50, 50, 60, 60]) == 0.0


def test_log_odds_compounds_weak_sensors():
    """The section 11 demo beat: the probability bar climbs as sensors stack."""
    thermal = [{"type": "thermal_hotspot", "conf": 0.71}]
    plus_rgb = thermal + [{"type": "rgb_person", "conf": 0.66, "bbox": [1, 1, 2, 2]}]
    plus_pulse = plus_rgb + [{"type": "rppg_pulse", "conf": 0.88}]
    plus_audio = plus_pulse + [{"type": "acoustic_distress", "conf": 0.54}]

    probabilities = [fuse_log_odds(d, ratios=RATIOS)[0]
                     for d in (thermal, plus_rgb, plus_pulse, plus_audio)]
    assert probabilities == sorted(probabilities), "adding evidence must not lower P"
    assert probabilities[0] < 0.5, "a thermal blob alone could be an engine block"
    assert probabilities[-1] > 0.9, "four agreeing sensors should be near certain"


def test_one_sensor_type_cannot_vote_twice():
    """Ten boxes from one camera are one witness, not ten."""
    one = [{"type": "rgb_person", "conf": 0.8, "bbox": [0, 0, 10, 10]}]
    many = [{"type": "rgb_person", "conf": 0.8, "bbox": [i, i, i + 10, i + 10]}
            for i in range(0, 100, 10)]
    assert fuse_log_odds(one, ratios=RATIOS)[0] == fuse_log_odds(many, ratios=RATIOS)[0]

    # Low-confidence noise must not drag the estimate up at all.
    weak = [{"type": "rgb_person", "conf": 0.05, "bbox": [0, 0, 10, 10]}]
    assert fuse_log_odds(weak, ratios=RATIOS)[0] == fuse_log_odds([], ratios=RATIOS)[0]

    # An unknown sensor type is ignored rather than silently trusted.
    assert sensor_key({"type": "psychic_vibes"}) is None
    assert fuse_log_odds([{"type": "psychic_vibes", "conf": 1.0}], ratios=RATIOS)[0] == \
        fuse_log_odds([], ratios=RATIOS)[0]


def test_merged_rgb_thermal_still_counts_both_sensors():
    merged = late_nms_merge(
        [{"type": "rgb_person", "conf": 0.66, "bbox": [100, 100, 150, 200]}],
        [{"type": "thermal_hotspot", "conf": 0.71, "bbox": [105, 98, 152, 205]}])
    _p, fired = fuse_log_odds(merged, ratios=RATIOS)
    assert {"rgb_person", "thermal_hotspot"} <= set(fired), \
        "merging boxes must not discard the thermal evidence"


def test_head_count_deduplicates_overlaps():
    two_people = [{"type": "rgb_person", "conf": 0.8, "bbox": [0, 0, 20, 40]},
                  {"type": "rgb_person", "conf": 0.7, "bbox": [200, 0, 220, 40]}]
    assert estimate_count(two_people) == 2
    same_person = [{"type": "rgb_person", "conf": 0.8, "bbox": [0, 0, 20, 40]},
                   {"type": "thermal_hotspot", "conf": 0.7, "bbox": [1, 2, 21, 41]}]
    assert estimate_count(same_person) == 1
    # A pulse or a shout with no box still means someone is there.
    assert estimate_count([{"type": "rppg_pulse", "conf": 0.9}]) == 1
    assert estimate_count([]) == 0


def test_fuse_cell_matches_the_contract():
    record = fuse_cell("12.9716_77.5946",
                       [{"type": "rgb_person", "conf": 0.66, "bbox": [120, 88, 164, 190]},
                        {"type": "rppg_pulse", "conf": 0.88, "bpm": 92}],
                       "drone_feed_03", timestamp="2026-09-12T08:41:12Z")
    assert {"cell_id", "source", "timestamp", "detections", "p_alive", "n_est"} <= set(record)
    assert 0.0 <= record["p_alive"] <= 1.0
    assert record["n_est"] >= 1


def test_gpu_guard_refuses_a_second_model():
    """4 GB means one model at a time, enforced rather than remembered."""
    assert active() is None
    with gpu_model("first", lambda: object()) as m:
        assert m is not None and active() == "first"
        try:
            with gpu_model("second", lambda: object()):
                raise AssertionError("loaded two GPU models at once")
        except RuntimeError:
            pass
    assert active() is None, "the guard must release even after a failed nested load"

    # A loader that raises must not leave the GPU marked busy forever.
    try:
        with gpu_model("broken", lambda: (_ for _ in ()).throw(ValueError("boom"))):
            pass
    except ValueError:
        pass
    assert active() is None


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
