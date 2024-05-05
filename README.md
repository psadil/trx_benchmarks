# Initial Benchmarks for trx configurations

This repo is designed to followup on these ongoing conversations about using tabular
file formats for trx: <https://github.com/tee-ar-ex/trx-python/discussions/63>

## To install

Benchmarks rely on data generously provided at: <https://usherbrooke-my.sharepoint.com/:f:/g/personal/rhef1902_usherbrooke_ca/Es5HfYK6fEpAg1o6wMbaQvEBjhRuX5lf-CjshwrVEZpQXg?e=ss5XXA>. Place that data in [data](data/)

A conda environment specification is in [env.yml](env.yml). A [parquet prototype](https://github.com/psadil/trx-parquet) exists but is not used, and instead the benchmarks rely on a bare-bones implementation in [trxparquet.py](trxparquet.py). This means that the trx-parquet
repo does _not_ need to be installed.

## To benchmark

Benchmarks are orchestrated with [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/index.html). To run and save the results as a json (in [.benchmarks](.benchmarks/)):

```shell
pytest --benchmark-autosave test_raw_io.py
```

For easier plotting, the resulting jsons can be converted into a csv.

```shell
pytest-benchmark compare --csv
```

## Existing Benchmarks

### raw_io

[test_raw_io.py](test_raw_io.py)

These benchmarks focus on the i/o for the building blocks of a trx file. The current implementation (that is, [trx-python](https://github.com/tee-ar-ex/trx-python)) writes and reads data with numpy. The proposed changes would leverage some format that is specific to tabular data, including [Apache Parquet](https://parquet.apache.org/) and [Feather](https://arrow.apache.org/docs/python/feather.html). Are there differences in speed when working with single vectors of data?

### raw_mmap

Numpy arrays, feather tables, and parquet files can all have a mmap to them. Are the substantive differences in how long these take to create?

### trx_io

There are many ways in which the core tools could be used to build a trx file. The reference implementation stores several arrays in a (possibly compressed) zip archive. That method is compared against a similar method with parquet (see `TrxParquet.to_file`), a method where all parquet tables are stored as separate tables in a directory (`TrxParquet.to_dir`), and approach where all data is stored in a single, big parquet file (`TrxParquet.to_table`), and one in which tables are gathered together as a duckdb (`TrxParquet.to_duckdb`).

### [TODO]

Additional tests will be added that focus on computation (e.g., differences in query speed, usability, code structure, etc).
