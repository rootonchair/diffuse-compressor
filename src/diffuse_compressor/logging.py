from __future__ import annotations

import json
import logging as py_logging
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import TextIO

from .artifact import ExportResult, QuantizedArtifact, QuantizedTarget
from .config import LoggingConfig


class _TeeStream:
    """Write stream output to the original stream and a log file."""

    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


class QuantizationLogger(py_logging.LoggerAdapter):
    """Logger used by quantization code, with optional run file outputs."""

    _instance: QuantizationLogger | None = None

    def __new__(cls, config: LoggingConfig | None = None, *, name: str = "diffuse_compressor") -> QuantizationLogger:
        del config, name
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: LoggingConfig | None = None, *, name: str = "diffuse_compressor") -> None:
        del name
        if not getattr(self, "_initialized", False):
            super().__init__(py_logging.getLogger("diffuse_compressor"), {})
            self.config: LoggingConfig | None = None
            self.text_path: Path | None = None
            self.target_records_path: Path | None = None
            self._text_file: TextIO | None = None
            self._stdout = None
            self._stderr = None
            self._file_handler: py_logging.Handler | None = None
            self._package_logger: py_logging.Logger | None = None
            self._package_logger_level: int | None = None
            self._target_elapsed: dict[str, list[float]] = {}
            self._active = False
            self._initialized = True
        if config is not None:
            self.configure(config)

    @classmethod
    def get_logger(cls, name: str) -> QuantizationLogger:
        """Return the default logger for a module."""

        del name
        return cls()

    def configure(self, config: LoggingConfig) -> None:
        """Configure the singleton for one quantization run."""

        self.config = config
        self.text_path = None
        self.target_records_path = None
        self._target_elapsed = {}

    @classmethod
    def is_enabled(cls, config: LoggingConfig | None) -> bool:
        """Return whether a logging config should create log files."""

        return config is not None and config.enabled and (config.text_output or config.target_records)

    def __enter__(self) -> QuantizationLogger:
        if self.config is None or not self.config.enabled:
            return self
        self._active = True
        self._resolve_paths()
        root = py_logging.getLogger()
        if self.config.text_output and self.text_path is not None:
            self._package_logger = py_logging.getLogger("diffuse_compressor")
            self._package_logger_level = self._package_logger.level
            if self._package_logger.level == py_logging.NOTSET or self._package_logger.level > py_logging.INFO:
                self._package_logger.setLevel(py_logging.INFO)
            self._text_file = self.text_path.open("a", encoding="utf-8")
            self._stdout = sys.stdout
            self._stderr = sys.stderr
            sys.stdout = _TeeStream(sys.stdout, self._text_file)  # type: ignore[assignment]
            sys.stderr = _TeeStream(sys.stderr, self._text_file)  # type: ignore[assignment]
            self._file_handler = py_logging.StreamHandler(self._text_file)
            self._file_handler.setFormatter(py_logging.Formatter("%(message)s"))
            root.addHandler(self._file_handler)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        root = py_logging.getLogger()
        if self._file_handler is not None:
            root.removeHandler(self._file_handler)
            self._file_handler.flush()
            self._file_handler = None
        if self._package_logger is not None and self._package_logger_level is not None:
            self._package_logger.setLevel(self._package_logger_level)
            self._package_logger = None
            self._package_logger_level = None
        if self._stdout is not None:
            sys.stdout = self._stdout
            self._stdout = None
        if self._stderr is not None:
            sys.stderr = self._stderr
            self._stderr = None
        if self._text_file is not None:
            self._text_file.flush()
            self._text_file.close()
            self._text_file = None
        self._active = False
        self.config = None

    def start_timing(self) -> float:
        """Return a timestamp for elapsed-time logging."""

        return time.perf_counter()

    def stop_timing(self, target: QuantizedTarget, started_at: float) -> None:
        """Record elapsed wall time for a completed target."""

        if not self._active or self.config is None or not self.config.enabled or not self.config.target_records:
            return
        elapsed_sec = time.perf_counter() - started_at
        self._target_elapsed.setdefault(target.target.export_name, []).append(elapsed_sec)

    def write_target_records(self, artifact: QuantizedArtifact, export_result: ExportResult | None = None) -> None:
        """Append one JSONL record for every quantized target."""

        if not self._active or self.config is None or not self.config.enabled or not self.config.target_records:
            return
        if self.target_records_path is None:
            self._resolve_paths()
        assert self.target_records_path is not None
        self.target_records_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = None if export_result is None else export_result.checkpoint_path
        with self.target_records_path.open("a", encoding="utf-8") as handle:
            for target in artifact.quantized_targets:
                handle.write(json.dumps(self._target_record(target, checkpoint_path), sort_keys=True) + "\n")

    def _target_record(self, target: QuantizedTarget, checkpoint_path: str | None) -> dict[str, object]:
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
