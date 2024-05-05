import pathlib
import shutil

DATADIR = pathlib.Path("data")


def cleanup_tmp(tmp_path: pathlib.Path) -> None:
    # cleanup after test
    for f in tmp_path.glob("*"):
        if f.is_file():
            f.unlink()
        else:
            shutil.rmtree(f)
