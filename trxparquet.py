import dataclasses
import json
import logging
import zipfile
from pathlib import Path

import duckdb
import nibabel as nb
import numpy as np
import numpy.typing as npt
import polars as pl
from trx import trx_file_memmap


def _write_frame(
    data: pl.DataFrame | pl.LazyFrame | None,
    f,
    **kwargs,
) -> None:
    """Write polars DataFrame or LazyFrame to parquet.

    Args:
        data (pl.DataFrame | pl.LazyFrame | None): DataFrame to write.
        f : Destination of parquet.
    """
    if data is None:
        return

    if isinstance(data, pl.DataFrame):
        data.write_parquet(f, **kwargs)
    else:
        data.sink_parquet(f, **kwargs)


def _write_frame_to_archive(
    af: zipfile.ZipFile,
    data: pl.DataFrame | pl.LazyFrame | None,
    arcname: str,
    **kwargs,
) -> None:
    """Write polars DataFrame or LazyFrame to location within archive file.

    Args:
        af (zipfile.ZipFile): Archive file.
        data (pl.DataFrame | pl.LazyFrame | None): Data to write.
        arcname (str): Location of new parquet file within archive.
    """
    if data is None:
        return

    with af.open(arcname, mode="w", force_zip64=True) as f:
        _write_frame(f=f, data=data, **kwargs)


def np_to_pl_dtype(dtype: np.dtype) -> pl.PolarsDataType:
    """Select polars type from numpy.

    Args:
        dtype (np.dtype): Numpy dtype to translate.

    Raises:
        ValueError: Error when not able to determine polars type

    Returns:
        pl.PolarsDataType: polars type for use in schema.
    """
    if dtype == np.float32:
        out = pl.Float32
    elif dtype == np.uint32:
        out = pl.UInt32
    elif dtype == np.uint16:
        out = pl.UInt16
    else:
        msg = "Unable to find matching dtype"
        raise ValueError(msg)

    return out


@dataclasses.dataclass(eq=True, frozen=True)
class TrxHeader:
    DIMENSIONS: npt.NDArray[np.uint16]
    VOXEL_TO_RASMM: npt.NDArray[np.float32]

    def to_bytes_dict(self) -> dict[bytes, bytes]:
        metadata = {
            b"DIMENSIONS": self.DIMENSIONS.tobytes(),
            b"VOXEL_TO_RASMM": self.VOXEL_TO_RASMM.tobytes(),
        }
        return metadata

    def to_dict(self) -> dict[str, list[int | list[float]]]:
        metadata = {
            "DIMENSIONS": self.DIMENSIONS.tolist(),
            "VOXEL_TO_RASMM": self.VOXEL_TO_RASMM.tolist(),
        }
        return metadata

    @classmethod
    def from_parquet_metadata(
        cls, metadata: dict[bytes, bytes]
    ) -> "TrxHeader":
        if _dimensions := metadata.get(b"DIMENSIONS"):
            dimensions = cls.parse_dimensions(_dimensions)
        else:
            msg = "Unable to find DIMENSIONS in metadata"
            raise AssertionError(msg)

        if _voxel_to_rasmm := metadata.get(b"VOXEL_TO_RASMM"):
            voxel_to_rasmm = cls.parse_voxel_to_rasmm(_voxel_to_rasmm)
        else:
            msg = "Unable to find VOXEL_TO_RASMM in metadata"
            raise AssertionError(msg)

        return cls(DIMENSIONS=dimensions, VOXEL_TO_RASMM=voxel_to_rasmm)

    @classmethod
    def from_nifti(
        cls, nifti: nb.nifti1.Nifti1Image | nb.nifti1.Nifti1Pair
    ) -> "TrxHeader":
        affine = nifti.affine
        if affine is None:
            msg = "Unable to find image affine"
            raise AssertionError(msg)
        return cls(
            DIMENSIONS=np.ndarray(nifti.shape, dtype=np.uint16),
            VOXEL_TO_RASMM=affine.astype(np.float32),
        )

    @staticmethod
    def parse_n_dps(n_dps: bytes) -> npt.NDArray[np.uint32]:
        return np.frombuffer(n_dps, dtype=np.uint32)

    @staticmethod
    def parse_n_dpv(n_dpb: bytes) -> npt.NDArray[np.uint32]:
        return np.frombuffer(n_dpb, dtype=np.uint32)

    @staticmethod
    def parse_dimensions(dimensions: bytes) -> npt.NDArray[np.uint16]:
        return np.frombuffer(dimensions, dtype=np.uint16)

    @staticmethod
    def parse_voxel_to_rasmm(voxel_to_rasmm: bytes) -> npt.NDArray[np.float32]:
        return np.frombuffer(voxel_to_rasmm, dtype=np.float32).reshape(4, 4)


@dataclasses.dataclass
class TrxParquet:
    """Barebones prototype for TrxParquet."""

    header: TrxHeader
    vertex: pl.DataFrame | pl.LazyFrame
    streamline: pl.DataFrame | pl.LazyFrame | None = None
    group: pl.DataFrame | pl.LazyFrame | None = None

    @classmethod
    def from_trx_file(cls, src: Path | str) -> "TrxParquet":
        """Build Parquet-style TrxParquet from .trx

        Args:
            src: Location of file to read.

        Returns:
            TrxParquet: Instance of TrxParquet
        """

        trxfile = trx_file_memmap.load(str(src))
        if trxfile.groups:
            logging.warning(
                "Detected groups in src, but groups not implemented. They will be ignored."
            )

        vertex_streamlines = pl.from_numpy(
            np.repeat(
                range(len(trxfile.streamlines)), trxfile.streamlines._lengths  # type: ignore
            ).astype(
                trxfile.streamlines._offsets.dtype  # type: ignore
            ),
            schema=["streamline_id"],
        )

        vertex_positions = pl.from_numpy(
            trxfile.streamlines._data,  # type: ignore
            schema=["0", "1", "2"],
        )

        _dps = []
        for k, v in trxfile.data_per_streamline.items():
            _dps.append(
                pl.DataFrame(
                    {
                        f"dps_{k}": pl.Series(
                            values=v,
                            dtype=pl.Array(
                                width=v.shape[1], inner=np_to_pl_dtype(v.dtype)
                            ),
                        )
                    }
                )
            )
        if len(_dps):
            dps = pl.concat(_dps, how="horizontal").hstack(
                vertex_streamlines.unique(
                    "streamline_id", maintain_order=True
                ).rename({"streamline_id": "id"})
            )
        else:
            dps = vertex_streamlines.unique(
                "streamline_id", maintain_order=True
            ).rename({"streamline_id": "id"})

        _dpv: list[pl.DataFrame] = []
        for k, v in trxfile.data_per_vertex.items():
            _dpv_dfs = []
            for streamline in v:
                _dpv_dfs.append(
                    pl.DataFrame(
                        pl.Series(
                            name=f"dpv_{k}",
                            dtype=pl.Array(
                                width=streamline.shape[1],
                                inner=np_to_pl_dtype(v[0].dtype),
                            ),
                            values=streamline,
                        )
                    )
                )

            _dpv.append(pl.concat(_dpv_dfs))

        if len(_dpv):
            dpv = (
                pl.concat(_dpv, how="horizontal")
                .with_columns(streamline_id=pl.Series(vertex_streamlines))
                .join(dps, on="protected_streamline")
                .hstack(vertex_positions)
            )
        else:
            dpv = vertex_positions.with_columns(
                streamline_id=pl.Series(vertex_streamlines)
            )

        metadata = {}
        metadata["DIMENSIONS"] = trxfile.header.get("DIMENSIONS")
        metadata["VOXEL_TO_RASMM"] = trxfile.header.get("VOXEL_TO_RASMM")
        header = TrxHeader(**metadata)

        return TrxParquet(header=header, streamline=dps, vertex=dpv)

    @classmethod
    def from_zip_file(cls, src: Path | str) -> "TrxParquet":
        """Read TrxParquet from (uncompressed) zip archive.

        Args:
            src : Location of zip archive to read.

        Returns:
            TrxParquet: Instance of TrxParquet.
        """
        with zipfile.ZipFile(src, mode="r") as zf:
            with zf.open("header.json") as f:
                header = json.load(f)
            with zf.open("streamline.parquet") as f:
                streamline = pl.read_parquet(f)
            with zf.open("vertex.parquet") as f:
                vertex = pl.read_parquet(f, columns=["0", "1", "2"])
            if "group.parquet" in zf.namelist():
                with zf.open("group.parquet") as f:
                    group = pl.read_parquet(f)
            else:
                group = None

        return cls(
            header=header, streamline=streamline, vertex=vertex, group=group
        )

    @classmethod
    def from_dir(cls, src: Path | str) -> "TrxParquet":
        """Instantiate TrxParquet from directory with header metadata and necessary tables.

        Args:
            src (Path | str): Path to directory containing header.json, streamline.parquet, and vertex.parquet.

        Returns:
            TrxParquet: Instance of TrxParquet.
        """
        _src = Path(src)
        header = json.loads((_src / "header.json").read_text())
        vertex = (
            pl.scan_parquet(_src / "vertex.parquet")
            .select("0", "1", "2")
            .collect()
        )
        streamline = pl.read_parquet(
            _src / "streamline.parquet", memory_map=False
        )
        if (g := _src / "group.parquet").exists():
            group = pl.read_parquet(g, memory_map=False)
        else:
            group = None

        return cls(
            header=header, streamline=streamline, vertex=vertex, group=group
        )

    @classmethod
    def from_parquet(cls, src: Path | str) -> "TrxParquet":
        """Instantiate TrxParquet from single table with vertex data (and associated header.json).

        Args:
            src (Path | str): Location of parquet table to store as vertex data.

        Returns:
            TrxParquet: Instance of TrxParquet.
        """
        _src = Path(src)
        header = json.loads(_src.with_suffix(".json").read_text())
        vertex = pl.read_parquet(_src, memory_map=False)
        streamline = None
        group = None
        return cls(
            header=header, streamline=streamline, vertex=vertex, group=group
        )

    @classmethod
    def from_duckdb(cls, src: Path | str) -> "TrxParquet":
        """Instantiate TrxParquet from duckdb file.

        Args:
            src (Path | str): Location of database to load.

        Returns:
            TrxParquet: Instance of TrxParquet.
        """
        _src = Path(src)
        header = json.loads(_src.with_suffix(".json").read_text())
        with duckdb.connect(str(_src)) as con:
            vertex = con.sql("SELECT x,y,z FROM vertex").pl()
            streamline = con.sql("SELECT * FROM streamline").pl()

        group = None
        return cls(
            header=header, streamline=streamline, vertex=vertex, group=group
        )

    def to_dir(
        self,
        dst: Path | str,
        **kwargs,
    ) -> Path:
        """Store TrxParquet as directory of tables (loadable by TrxParquet.from_dir)

        Args:
            dst (Path | str): Directory in which files will be stored. Must not exist.
            kwargs: Named arguments passed on to polars.DataFrame.write_parquet or polars.LazyFrame.sink_parquet

        Returns:
            Path: Path where files were written.
        """
        _dst = Path(dst)
        _dst.mkdir(parents=True)

        (_dst / "header.json").write_text(json.dumps(self.header.to_dict()))
        _write_frame(f=_dst / "vertex.parquet", data=self.vertex, **kwargs)
        _write_frame(
            f=_dst / "streamline.parquet", data=self.streamline, **kwargs
        )
        _write_frame(f=_dst / "group.parquet", data=self.group, **kwargs)
        return _dst

    def to_file(self, dst: Path | str, **kwargs) -> Path:
        """Write to zip archive.

        Args:
            dst (Path | str): Archive to create.
            kwargs: Named arguments passed on to polars.DataFrame.write_parquet or polars.LazyFrame.sink_parquet

        Returns:
            Path: Location of newly written archive.
        """

        with zipfile.ZipFile(
            dst, mode="w", compression=zipfile.ZIP_STORED
        ) as zf:
            zf.writestr(
                "header.json",
                data=json.dumps(self.header.to_dict()),
            )
            _write_frame_to_archive(
                zf, self.streamline, arcname="streamline.parquet", **kwargs
            )
            _write_frame_to_archive(
                zf, self.vertex, arcname="vertex.parquet", **kwargs
            )
            _write_frame_to_archive(
                zf, self.group, arcname="group.parquet", **kwargs
            )

        return Path(dst)

    def to_parquet(self, dst: Path | str, **kwargs) -> Path:
        """Store TrxParquet in single parquet table (vertex table).

        Args:
            dst (Path | str): Name of parquet file to write.
            kwargs: Named arguments passed on to polars.DataFrame.write_parquet

        Returns:
            Path: Path to written file.
        """
        assert isinstance(self.streamline, pl.DataFrame)
        assert isinstance(self.vertex, pl.DataFrame)
        _dst = Path(dst)
        tbl = self.vertex.join(
            self.streamline,
            how="left",
            left_on="streamline_id",
            right_on="id",
            validate="m:1",
        )
        tbl.write_parquet(_dst, **kwargs)
        _dst.with_suffix(".json").write_text(json.dumps(self.header.to_dict()))
        return _dst

    def to_duckdb(self, dst: Path | str) -> Path:
        """Store TrxParquet in duckdb file.

        Args:
            dst (Path | str): Name of database file to write.

        Returns:
            Path: Path to database.
        """
        _dst = Path(dst)
        s = self.streamline  # noqa: F841
        v = self.vertex  # noqa: F841
        with duckdb.connect(str(dst)) as con:
            con.sql("CREATE TABLE streamline (id UINTEGER PRIMARY KEY)")
            con.sql(
                """CREATE TABLE vertex (
                    x FLOAT4,
                    y FLOAT4,
                    z FLOAT4,
                    streamline_id UINTEGER,
                    FOREIGN KEY (streamline_id) REFERENCES streamline (id)
                    )
                """
            )
            con.sql("INSERT INTO streamline SELECT * from s")
            con.sql("INSERT INTO vertex SELECT * from v")

        _dst.with_suffix(".json").write_text(json.dumps(self.header.to_dict()))
        return _dst
