import pathlib
import typing

import numpy as np
import polars as pl
import pytest
from pyarrow import feather

import utils

# pytest-benchmark compare --group-by=group,param:n


def _gen_array(n: int) -> np.ndarray:
    rng = np.random.default_rng(12345)
    return rng.normal(size=(n, 3))


def _get_bench_write(
    data: np.ndarray, tmp_path: pathlib.Path, filetype: str
) -> typing.Callable:
    match filetype:
        case "parquet":
            df = pl.from_numpy(data)

            def to_bench():
                df.write_parquet(
                    file=tmp_path / "data.parquet", compression="uncompressed"
                )

        case "numpy":

            def to_bench():
                data.tofile(tmp_path / "data.npy")

        case "arrow":
            df = pl.from_numpy(data)

            def to_bench():
                df.write_ipc(file=tmp_path / "data.arrow")

        case "feather":
            df = pl.from_numpy(data).to_arrow()

            def to_bench():
                feather.write_feather(
                    df,
                    dest=tmp_path / "data.feather",
                    compression="uncompressed",
                )

        case _:
            raise AssertionError

    return to_bench


def _get_bench_read(
    data: np.ndarray, tmp_path: pathlib.Path, filetype: str
) -> typing.Callable:
    match filetype:
        case "parquet":
            out = tmp_path / "data.parquet"
            pl.from_numpy(data).write_parquet(out, compression="uncompressed")

            def to_bench():
                pl.read_parquet(out, memory_map=False)

        case "numpy":
            dtype = data.dtype
            out = tmp_path / "data.npy"
            data.tofile(out)

            def to_bench():
                np.fromfile(file=out, dtype=dtype)

        case "arrow":
            out = tmp_path / "data.arrow"
            pl.from_numpy(data).write_ipc(out, future=True)

            def to_bench():
                pl.read_ipc(source=out, memory_map=False)

        case "feather":
            out = tmp_path / "data.feather"
            feather.write_feather(
                pl.from_numpy(data).to_arrow(),
                dest=out,
                compression="uncompressed",
            )

            def to_bench():
                feather.read_feather(source=out)

        case _:
            raise AssertionError

    return to_bench


@pytest.mark.benchmark(group=("write raw"))
@pytest.mark.parametrize("filetype", ["numpy", "parquet", "arrow", "feather"])
@pytest.mark.parametrize("n", [1000000, 10000000, 100000000])
def test_write(tmp_path: pathlib.Path, benchmark, filetype: str, n: int):
    data = _gen_array(n=n)
    to_bench = _get_bench_write(
        filetype=filetype, tmp_path=tmp_path, data=data
    )
    benchmark(to_bench)
    utils.cleanup_tmp(tmp_path)


@pytest.mark.benchmark(group=("read raw"))
@pytest.mark.parametrize("filetype", ["numpy", "parquet", "arrow", "feather"])
@pytest.mark.parametrize("n", [1000000, 10000000, 100000000])
def test_read(tmp_path: pathlib.Path, benchmark, filetype: str, n: int):
    data = _gen_array(n=n)
    to_bench = _get_bench_read(filetype=filetype, tmp_path=tmp_path, data=data)
    benchmark(to_bench)
    utils.cleanup_tmp(tmp_path)
