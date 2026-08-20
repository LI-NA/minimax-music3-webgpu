"""ONNX external-data repacking with bounded artifact files."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import onnx
from onnx import numpy_helper


@dataclass(frozen=True)
class ExternalDataShard:
    path: Path
    size: int


@dataclass(frozen=True)
class RepackedModel:
    model_path: Path
    shards: tuple[ExternalDataShard, ...]


@dataclass(frozen=True)
class _TensorSource:
    tensor: onnx.TensorProto
    path: Path | None
    offset: int
    length: int
    data: bytes | None


def repack_external_data(
    model_path: Path, output_dir: Path, max_file_bytes: int, inline_threshold: int = 0
) -> RepackedModel:
    model = onnx.load_model(model_path, load_external_data=False)
    sources = [_tensor_source(tensor, model_path.parent) for tensor in model.graph.initializer]
    for source in sources:
        if source.length > max_file_bytes:
            raise ValueError(f"initializer {source.tensor.name} exceeds artifact limit")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_dir.name}-staging-", dir=output_dir.parent) as staging:
        staging_dir = Path(staging)
        generation = f"weights-{uuid4().hex}"
        shards = _write_staged_shards(sources, staging_dir, max_file_bytes, generation, inline_threshold)
        staged_model = staging_dir / model_path.name
        onnx.save_model(model, staged_model)
        _validate_ranges(staged_model, max_file_bytes)
        output_dir.mkdir(parents=True, exist_ok=True)
        packed_path = output_dir / model_path.name
        promoted = []
        try:
            for shard in shards:
                target = output_dir / shard.path.name
                promoted.append(target)
                _promote_shard(shard.path, target)
            staged_model.replace(packed_path)
        except Exception:
            for path in promoted:
                path.unlink(missing_ok=True)
            raise
        return RepackedModel(
            model_path=packed_path,
            shards=tuple(
                ExternalDataShard(output_dir / shard.path.name, shard.size) for shard in shards
            ),
        )


def _tensor_source(tensor: onnx.TensorProto, model_dir: Path) -> _TensorSource:
    if tensor.data_location != onnx.TensorProto.EXTERNAL:
        data = tensor.raw_data or numpy_helper.to_array(tensor).tobytes()
        return _TensorSource(tensor, None, 0, len(data), data)

    fields = {entry.key: entry.value for entry in tensor.external_data}
    if "location" not in fields:
        raise ValueError(f"initializer {tensor.name} has no external-data location")
    source_path = _source_path(model_dir, fields["location"])
    offset = int(fields.get("offset", 0))
    file_size = source_path.stat().st_size
    length = int(fields["length"]) if "length" in fields else file_size - offset
    if offset < 0 or length < 0 or offset + length > file_size:
        raise ValueError(f"initializer {tensor.name} has an invalid source range")
    return _TensorSource(tensor, source_path, offset, length, None)


def _source_path(model_dir: Path, location: str) -> Path:
    path = Path(location)
    if path.is_absolute():
        raise ValueError("external-data location must be relative")
    source_root = model_dir.resolve()
    resolved = (source_root / path).resolve()
    if not resolved.is_relative_to(source_root):
        raise ValueError("external-data location escapes the model directory")
    return resolved


def _write_staged_shards(
    sources: list[_TensorSource], staging_dir: Path, max_file_bytes: int, generation: str,
    inline_threshold: int,
) -> list[ExternalDataShard]:
    shards: list[ExternalDataShard] = []
    current_file = None
    current_size = 0
    try:
        for source in sources:
            if source.length <= inline_threshold:
                _set_inline_data(source)
                continue
            if current_file is None or current_size + source.length > max_file_bytes:
                path = staging_dir / f"{generation}-{len(shards):03d}.bin"
                current_file = path.open("wb")
                shards.append(ExternalDataShard(path, 0))
                current_size = 0
            _copy_source(source, current_file)
            _set_external_fields(source.tensor, Path(current_file.name).name, current_size, source.length)
            current_size += source.length
    finally:
        if current_file is not None:
            current_file.close()
    return [ExternalDataShard(shard.path, shard.path.stat().st_size) for shard in shards]


def _set_inline_data(source: _TensorSource) -> None:
    if source.data is not None:
        data = source.data
    else:
        with source.path.open("rb") as file:
            file.seek(source.offset)
            data = file.read(source.length)
        if len(data) != source.length:
            raise ValueError("external data is shorter than initializer metadata")
    del source.tensor.external_data[:]
    source.tensor.data_location = onnx.TensorProto.DEFAULT
    source.tensor.raw_data = data


def _promote_shard(staged_path: Path, target: Path) -> None:
    staged_path.replace(target)


def _copy_source(source: _TensorSource, destination) -> None:
    if source.data is not None:
        destination.write(source.data)
        return
    with source.path.open("rb") as file:
        file.seek(source.offset)
        _copy_exact(file, destination, source.length)


def _set_external_fields(tensor, location: str, offset: int, length: int) -> None:
    del tensor.external_data[:]
    for key, value in (("location", location), ("offset", offset), ("length", length)):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = str(value)
    for field in (
        "raw_data", "float_data", "int32_data", "string_data", "int64_data",
        "double_data", "uint64_data",
    ):
        tensor.ClearField(field)
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
        fields = {entry.key: entry.value for entry in tensor.external_data}
        if not fields:
            continue
        location = fields["location"]
        offset = int(fields["offset"])
        length = int(fields["length"])
        shard = _source_path(model_path.parent, location)
        if offset + length > max_file_bytes or offset + length > shard.stat().st_size:
            raise ValueError(f"initializer {tensor.name} has an invalid external-data range")
