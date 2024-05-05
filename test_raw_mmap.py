import pathlib
import typing

import numpy as np
import polars as pl
import pytest

import utils

# pytest --benchmark-group-by=group,param:n test_raw_mmap.py


def _gen_array(n: int) -> np.ndarray:
    rng = np.random.default_rng(12345)
    return rng.normal(size=(n,))


def _get_bench_read(
    data: np.ndarray, tmp_path: pathlib.Path, filetype: str
) -> typing.Callable:
    match filetype:
        case "parquet":
            out = tmp_path / "data.parquet"
            pl.from_numpy(data).write_parquet(out, compression="uncompressed")

            def to_bench():
                pl.read_parquet(out, memory_map=True)

        case "numpy":
            dtype = data.dtype
            shape = data.shape
            out = tmp_path / "data.npy"
            data.tofile(out)

            def to_bench():
                np.memmap(filename=out, dtype=dtype, shape=shape)

        case "arrow":
            out = tmp_path / "data.arrow"
            pl.from_numpy(data).write_ipc(out, future=True)

            def to_bench():
                pl.read_ipc(source=out, memory_map=True)

        case _:
            raise AssertionError

    return to_bench


@pytest.mark.benchmark(group=("mmap raw"))
@pytest.mark.parametrize("filetype", ["numpy", "parquet", "arrow"])
@pytest.mark.parametrize("n", [1000000, 100000000])
def test_mmap(tmp_path: pathlib.Path, benchmark, filetype: str, n: int):
    data = _gen_array(n=n)
    to_bench = _get_bench_read(filetype=filetype, tmp_path=tmp_path, data=data)
    benchmark(to_bench)
    utils.cleanup_tmp(tmp_path)
