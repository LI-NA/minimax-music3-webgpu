"""ONNX external-data repacking with bounded artifact files."""

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import onnx


@dataclass(frozen=True)
class ExternalDataShard:
    path: Path
    size: int


@dataclass(frozen=True)
class RepackedModel:
    model_path: Path
    shards: tuple[ExternalDataShard, ...]


def repack_external_data(
    model_path: Path, output_dir: Path, max_file_bytes: int
) -> RepackedModel:
    model = onnx.load_model(model_path, load_external_data=False)
    tensors = [(tensor, _external_fields(tensor)) for tensor in model.graph.initializer]
    for tensor, (_, _, length) in tensors:
        if length > max_file_bytes:
            raise ValueError(f"initializer {tensor.name} exceeds artifact limit")

    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[tuple[Path, object]] = []
    current_file = None
    current_size = 0
    try:
        for tensor, (location, offset, length) in tensors:
            if current_file is None or current_size + length > max_file_bytes:
                path = output_dir / f"weights-{len(files):03d}.bin"
                current_file = path.open("wb")
                files.append((path, current_file))
                current_size = 0
            source_path = (model_path.parent / location).resolve()
            with source_path.open("rb") as source:
                source.seek(offset)
                _copy_exact(source, current_file, length)
            _set_external_fields(tensor, current_file.name, current_size, length)
            current_size += length
    finally:
        for _, file in files:
            file.close()

    packed_path = output_dir / model_path.name
    temporary_path = output_dir / f".{model_path.name}.{uuid4().hex}.tmp"
    onnx.save_model(model, temporary_path)
    temporary_path.replace(packed_path)
    _validate_ranges(packed_path, max_file_bytes)
    return RepackedModel(
        model_path=packed_path,
        shards=tuple(ExternalDataShard(path, path.stat().st_size) for path, _ in files),
    )


def _external_fields(tensor) -> tuple[str, int, int]:
    fields = {entry.key: entry.value for entry in tensor.external_data}
    if "location" not in fields:
        raise ValueError(f"initializer {tensor.name} is not external data")
    return fields["location"], int(fields.get("offset", 0)), int(fields["length"])


def _set_external_fields(tensor, location: str, offset: int, length: int) -> None:
    del tensor.external_data[:]
    for key, value in (("location", location), ("offset", offset), ("length", length)):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = str(value)
    tensor.data_location = onnx.TensorProto.EXTERNAL


def _copy_exact(source, destination, length: int) -> None:
    remaining = length
    while remaining:
        chunk = source.read(min(1024 * 1024, remaining))
        if not chunk:
            raise ValueError("external data is shorter than initializer metadata")
        destination.write(chunk)
        remaining -= len(chunk)


def _validate_ranges(model_path: Path, max_file_bytes: int) -> None:
    model = onnx.load_model(model_path, load_external_data=False)
    for tensor in model.graph.initializer:
        location, offset, length = _external_fields(tensor)
        shard = model_path.parent / location
        if offset + length > max_file_bytes or offset + length > shard.stat().st_size:
            raise ValueError(f"initializer {tensor.name} has an invalid external-data range")
