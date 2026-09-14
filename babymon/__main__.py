"""Command line entry point.

    python -m babymon --config config.yaml
    python -m babymon --config config.yaml --source clip.avi --replay
"""

from __future__ import annotations

import argparse
import logging
import sys

from babymon.config import Config
from babymon.events import Alert
from babymon.detectors.motion import MotionDetector
from babymon.detectors.person import PersonDetector, YoloPersonModel
from babymon.rules.engine import RuleEngine
from babymon.rules.zones import Zone
from babymon.runner import Pipeline, run_replay
from babymon.sinks.store import SqliteStore
from babymon.sources.file import FileVideoSource
from babymon.sources.webcam import WebcamSource


def format_alert(alert: Alert) -> str:
    """One line per alert, for the replay listing.

    The first column is when the CONDITION began, not when the alert fired.
    For ``awake`` those are nearly the same; for ``lost_track`` they are not —
    it began when the baby was last seen and fired a minute later. Printing
    the bare number under an unlabelled column read as "the alert happened at
    0.00s", which is wrong and alarming. The detail column carries the number
    a person actually wants.
    """
    md = alert.metadata
    if "seconds_since_last_seen" in md:
        detail = f"baby not seen for {md['seconds_since_last_seen']}s"
        if md.get("adult_present"):
            detail += " (an adult is present)"
    elif "detector" in md:
        detail = f"detector {md['detector']!r} stopped reporting"
    elif "motion_energy" in md:
        detail = f"motion energy {md['motion_energy']:.3f}"
    elif "reason" in md:
        detail = str(md["reason"])
    else:
        detail = ""
    return (
        f"{alert.started_at:8.2f}s  {alert.severity.value:8}  "
        f"{alert.type.value:16}  {detail}".rstrip()
    )


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

    cfg = Config.load(args.config, source=args.source)

    if args.replay and cfg.source.kind != "file":
        print(
            "--replay requires a finite source; pass --source <path to a "
            "video file> (or set source.kind: file in the config)",
            file=sys.stderr,
        )
        return 1

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
        if alerts:
            print(f"{'began':>8}  {'severity':8}  {'event':16}  detail")
        for alert in alerts:
            print(format_alert(alert))
        store.close()
        return 0

    # The exit code is not decoration: a monitor that died because its
    # source became unreadable must not report success, or a supervisor set
    # to Restart=on-failure will leave the cot unwatched.
    return Pipeline(
        source=source,
        detectors=detectors,
        engine=engine,
        sinks=[store],
        strides={
            "motion": cfg.detectors.motion_stride,
            "person": cfg.detectors.person_stride,
        },
    ).run()


if __name__ == "__main__":
    sys.exit(main())
