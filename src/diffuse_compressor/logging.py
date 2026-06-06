from __future__ import annotations

import json
import logging as py_logging
import time
from pathlib import Path
from typing import Any

from .artifact import ExportResult, QuantizedArtifact, QuantizedTarget
from .config import LoggingConfig


_PACKAGE_LOGGER_NAME = "diffuse_compressor"


class QuantizationLogger:
    """Thin wrapper around stdlib logging plus optional quantization run outputs."""

    def __init__(
        self,
        name: str | py_logging.Logger | LoggingConfig = _PACKAGE_LOGGER_NAME,
        config: LoggingConfig | None = None,
    ) -> None:
        if isinstance(name, LoggingConfig):
            config = name if config is None else config
            name = _PACKAGE_LOGGER_NAME
        self.logger = (
            name if isinstance(name, py_logging.Logger) else py_logging.getLogger(name)
        )
        self.config = config
        self.text_path: Path | None = None
        self.target_records_path: Path | None = None
        self._target_elapsed: dict[str, list[float]] = {}
        if self.enabled:
            self._resolve_paths()

    @classmethod
    def get_logger(cls, name: str) -> QuantizationLogger:
        return cls(name)

    @property
    def enabled(self) -> bool:
        return (
            self.config is not None
            and self.config.enabled
            and (self.config.text_output or self.config.target_records)
        )

    @property
    def text_output(self) -> bool:
        return (
            self.enabled
            and self.config is not None
            and self.config.text_output
            and self.text_path is not None
        )

    @property
    def target_records(self) -> bool:
        return (
            self.enabled
            and self.config is not None
            and self.config.target_records
            and self.target_records_path is not None
        )

    def debug(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.logger.debug(msg, *args, **kwargs)
        self._write_log_text(py_logging.DEBUG, msg, *args)

    def info(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.logger.info(msg, *args, **kwargs)
        self._write_log_text(py_logging.INFO, msg, *args)

    def warning(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.logger.warning(msg, *args, **kwargs)
        self._write_log_text(py_logging.WARNING, msg, *args)

    def error(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.logger.error(msg, *args, **kwargs)
        self._write_log_text(py_logging.ERROR, msg, *args)

    def exception(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.logger.exception(msg, *args, **kwargs)
        self._write_log_text(py_logging.ERROR, msg, *args)

    def isEnabledFor(self, level: int) -> bool:
        return self.logger.isEnabledFor(level)

    def start_timing(self) -> float:
        return time.perf_counter()

    def stop_timing(self, target: QuantizedTarget, started_at: float) -> None:
        if not self.target_records:
            return
        elapsed_sec = time.perf_counter() - started_at
        self._target_elapsed.setdefault(target.target.export_name, []).append(
            elapsed_sec
        )

    def write_target_records(
        self, artifact: QuantizedArtifact, export_result: ExportResult | None = None
    ) -> None:
        if not self.target_records:
            return
        assert self.target_records_path is not None
        self.target_records_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = (
            None if export_result is None else export_result.checkpoint_path
        )
        with self.target_records_path.open("a", encoding="utf-8") as handle:
            for target in artifact.quantized_targets:
                handle.write(
                    json.dumps(
                        self._target_record(target, checkpoint_path), sort_keys=True
                    )
                    + "\n"
                )

    def _write_log_text(self, level: int, msg: object, *args: object) -> None:
        if not self.text_output or level < py_logging.INFO:
            return
        assert self.text_path is not None
        self.text_path.parent.mkdir(parents=True, exist_ok=True)
        text = str(msg)
        if args:
            try:
                text = text % args
            except TypeError:
                text = " ".join([text, *(str(arg) for arg in args)])
        with self.text_path.open("a", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")

    def _target_record(
        self, target: QuantizedTarget, checkpoint_path: str | None
    ) -> dict[str, object]:
        low_rank = target.metadata.get("low_rank_solver", {})
        low_rank = low_rank if isinstance(low_rank, dict) else {}
        return {
            "target": target.target.export_name,
            "modules": list(target.target.module_names),
            "precision": target.metadata.get("precision"),
            "group_size": target.metadata.get("group_size"),
            "rank": target.metadata.get("rank"),
            "elapsed_sec": self._elapsed_for(target.target.export_name),
            "low_rank_mode": low_rank.get("mode"),
            "best_error": low_rank.get("best_error"),
            "errors": low_rank.get("errors"),
            "iterations": low_rank.get("iterations"),
            "stopped_early": low_rank.get("stopped_early"),
            "calibrated": target.metadata.get("calibrated"),
            "checkpoint_path": checkpoint_path,
        }

    def _elapsed_for(self, target: str) -> float | None:
        records = self._target_elapsed.get(target)
        if not records:
            return None
        return records.pop(0)

    def _resolve_paths(self) -> None:
        if self.config is None:
            return
        if self.text_path is not None or self.target_records_path is not None:
            return
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        base = Path(self.config.name or "quantization").stem or "quantization"
        text_path, records_path = self._available_paths(log_dir, base)
        if self.config.text_output:
            self.text_path = text_path
        if self.config.target_records:
            self.target_records_path = records_path

    @staticmethod
    def _available_paths(log_dir: Path, base: str) -> tuple[Path, Path]:
        text_path = log_dir / f"{base}.txt"
        records_path = log_dir / f"{base}.targets.jsonl"
        if not text_path.exists() and not records_path.exists():
            return text_path, records_path
        suffix = time.strftime("%Y%m%d-%H%M%S")
        candidate_text = log_dir / f"{base}-{suffix}.txt"
        candidate_records = log_dir / f"{base}-{suffix}.targets.jsonl"
        index = 1
        while candidate_text.exists() or candidate_records.exists():
            candidate_text = log_dir / f"{base}-{suffix}-{index}.txt"
            candidate_records = log_dir / f"{base}-{suffix}-{index}.targets.jsonl"
            index += 1
        return candidate_text, candidate_records
