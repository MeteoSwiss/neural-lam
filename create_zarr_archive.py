# Standard library
import argparse
import os

# Third-party
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from numcodecs import Blosc


def process_grib_files(data_in, selected_vars, indexpath):
    """Process grib2 files and return a dataset."""
    datasets = []
    for filters, variables in selected_vars.items():
        backend_kwargs = {
            "errors": "ignore",
            "encode_cf": ["time", "geography"],
            "indexpath": indexpath + "{path}.{short_hash}.idx",
            "filter_by_keys": {},
        }
        print(f"Processing {filters} with variables {variables}")
        if filters == "shortName":
            for var in variables:
                backend_kwargs["filter_by_keys"] = {"shortName": var}
                with ProgressBar():
                    ds = xr.open_mfdataset(
                        data_in,
                        engine="cfgrib",
                        backend_kwargs=backend_kwargs,
                        combine="nested",
                        concat_dim="time",
                        parallel=True,
                        chunks={"time": 1, "level": 1},
                        data_vars="minimal",
                        coords="minimal",
                        compat="override",
                    )
                datasets.append(ds)
        else:
            backend_kwargs["filter_by_keys"] = {"typeOfLevel": filters}
            if filters == "surface":
                backend_kwargs["filter_by_keys"]["stepType"] = "instant"
            with ProgressBar():
                ds = xr.open_mfdataset(
                    data_in,
                    engine="cfgrib",
                    backend_kwargs=backend_kwargs,
                    combine="nested",
                    concat_dim="time",
                    parallel=True,
                    chunks={"time": 1},
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                )
            datasets.append(ds[vars])
    return xr.merge(datasets, compat="minimal")


def main(data_in, data_out, indexpath, selected_vars, selected_vars_2):
    """Process grib2 files and save them to zarr format."""
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    all_files = [
        os.path.join(root, f)
        for root, dirs, files in os.walk(data_in)
        for f in files
        if root.endswith("det") and f.startswith("laf")
    ]

    print("Processing training data_out")
    ds = xr.merge(
        [
            process_grib_files(all_files, selected_vars, indexpath),
            process_grib_files(all_files, selected_vars_2, indexpath),
        ],
        compat="override",
    ).rename_vars({"pp": "PP"})

    print(f"Saving Zarr to {data_out}")
    with ProgressBar():
        ds = ds.assign_coords(x=ds.x, y=ds.y).drop_vars(["valid_time", "step"])
        ds.to_zarr(
            data_out,
            consolidated=True,
            mode="w",
            encoding={var: {"compressor": compressor} for var in ds.data_vars},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process grib2 files and save them to zarr format."
    )
    parser.add_argument(
        "--data_in",
        type=str,
        default="/scratch/mch/dealmeih/kenda/",
        help="Input data_out directory.",
    )
    parser.add_argument(
        "--data_out",
        type=str,
        default="/scratch/mch/sadamov/data.zarr",
        help="Output zarr file for training.",
    )
    parser.add_argument(
        "--indexpath",
        type=str,
        default="/scratch/mch/sadamov/temp",
        help="Path to the index file.",
    )

    SELECTED_VARS = {
        "heightAboveGround": ["U_10M", "V_10M"],
        "surface": ["PS", "HSURF", "TQV", "FIS", "CLCL"],
        "hybridLayer": ["T", "pp", "QV"],
        "hybrid": ["W"],
        "meanSea": ["PMSL"],
    }
    SELECTED_VARS_2 = {
        # U,V have different lat/lon. T_2M has different heightAboveGround
        "shortName": ["T_2M", "U", "V"],
    }

    args = parser.parse_args()

    cluster = SLURMCluster(
        queue="postproc",
        account="s83",
        processes=1,
        cores=256,
        memory="446GB",
        local_directory="/scratch/mch/sadamov/temp",
        shared_temp_directory="/scratch/mch/sadamov/temp",
        log_directory="lightning_logs",
        shebang="#!/bin/bash",
        interface="nmn0",
        walltime="5-00:00:00",
        job_extra_directives=["--exclusive"],
    )
    cluster.scale(jobs=2)
    client = Client(cluster)
    client.wait_for_workers(2)

    main(
        args.data_in,
        args.data_out,
        args.indexpath,
        SELECTED_VARS,
        SELECTED_VARS_2,
    )

    client.close()
