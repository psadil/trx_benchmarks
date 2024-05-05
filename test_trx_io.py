import pathlib
import typing

import pytest
from trx import trx_file_memmap

import trxparquet
import utils


def _get_bench_read(
    tmp_path: pathlib.Path, src: pathlib.Path, filetype: str
) -> typing.Callable:
    match filetype:
        case "trxfile":

            def to_bench():
                trx_file_memmap.load(src).to_memory()  # type: ignore

        case "trxparquet_zip":
            tpf = trxparquet.TrxParquet.from_trx_file(src)
            to_read = tmp_path / "test.zip"
            tpf.to_file(to_read, compression="uncompressed")

            def to_bench():
                trxparquet.TrxParquet.from_zip_file(to_read)

        case "trxparquet_dir":
            tpf = trxparquet.TrxParquet.from_trx_file(src)
            to_read = tmp_path / "test"
            tpf.to_dir(to_read, compression="uncompressed")

            def to_bench():
                trxparquet.TrxParquet.from_dir(to_read)

        case "trxparquet_tbl":
            tpf = trxparquet.TrxParquet.from_trx_file(src)
            to_read = tmp_path / "test.parquet"
            tpf.to_parquet(to_read, compression="uncompressed")

            def to_bench():
                trxparquet.TrxParquet.from_parquet(to_read)

        case "trxparquet_duck":
            tpf = trxparquet.TrxParquet.from_trx_file(src)
            to_read = tmp_path / "test.db"
            tpf.to_duckdb(to_read)

            def to_bench():
                trxparquet.TrxParquet.from_duckdb(to_read)

        case _:
            raise AssertionError("Unable to handle filetype")

    return to_bench


@pytest.mark.benchmark(group=("read trx"))
@pytest.mark.parametrize(
    "filetype",
    ["trxfile", "trxparquet_dir", "trxparquet_tbl", "trxparquet_zip"],
)
@pytest.mark.parametrize("src", ["f32_ui32_wo_metadata.trx"])
def test_read(tmp_path: pathlib.Path, benchmark, src: str, filetype: str):
    to_bench = _get_bench_read(
        filetype=filetype, tmp_path=tmp_path, src=utils.DATADIR / src
    )
    benchmark(to_bench)
    utils.cleanup_tmp(tmp_path)
