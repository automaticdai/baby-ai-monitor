"""Command line entry point.

    python -m babymon --config config.yaml
    python -m babymon --config config.yaml --source clip.avi --replay
"""

from __future__ import annotations

import argparse
import logging
import sys

from babymon.config import Config
from babymon.detectors.motion import MotionDetector
from babymon.detectors.person import PersonDetector, YoloPersonModel
from babymon.rules.engine import RuleEngine
from babymon.rules.zones import Zone
from babymon.runner import Pipeline, run_replay
from babymon.sinks.store import SqliteStore
from babymon.sources.file import FileVideoSource
from babymon.sources.webcam import WebcamSource


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="babymon")
    p.add_argument("--config", help="path to YAML config file")
    p.add_argument("--source", help="video file to use instead of the webcam")
    p.add_argument(
        "--replay",
        action="store_true",
        help="process the file as fast as possible and exit",
    )
    p.add_argument("--no-person", action="store_true", help="skip YOLO detection")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = Config.load(args.config) if args.config else Config()
    if args.source:
        cfg.source.kind = "file"
        cfg.source.path = args.source

    crib_cfg = cfg.zone("crib")
    crib = Zone.from_config(crib_cfg) if crib_cfg else None
    if crib is None:
        logging.warning(
            "no 'crib' zone configured: zone exit and baby/adult attribution "
            "are disabled"
        )

    detectors = [
        MotionDetector(
            zone=crib, pixel_threshold=cfg.detectors.motion_pixel_threshold
        )
    ]
    if not args.no_person:
        detectors.append(
            PersonDetector(
                model=YoloPersonModel(cfg.detectors.model_path),
                crib_zone=crib,
                conf_threshold=cfg.detectors.person_conf_threshold,
                baby_max_area=cfg.detectors.baby_max_area,
            )
        )

    engine = RuleEngine(
        config=cfg.rules,
        crib_zone=crib,
        watched_detectors=tuple(d.name for d in detectors),
    )
    store = SqliteStore(
        db_path=cfg.storage.db_path,
        snapshot_dir=cfg.storage.snapshot_dir,
        retention_days=cfg.storage.retention_days,
    )
    store.sweep()

    if cfg.source.kind == "file":
        source = FileVideoSource(
            cfg.source.path,
            source_id=cfg.source.source_id,
            realtime=not args.replay,
        )
    else:
        source = WebcamSource(
            device=cfg.source.device, source_id=cfg.source.source_id
        )

    if args.replay:
        alerts = run_replay(source, detectors, engine, [store])
        for alert in alerts:
            print(f"{alert.started_at:8.2f}s  {alert.severity.value:8}  {alert.type.value}")
        store.close()
        return 0

    Pipeline(
        source=source,
        detectors=detectors,
        engine=engine,
        sinks=[store],
        strides={
            "motion": cfg.detectors.motion_stride,
            "person": cfg.detectors.person_stride,
        },
    ).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
