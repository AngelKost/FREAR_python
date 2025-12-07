import pickle
import os

import pandas as pd

from typing import Any, Dict

from frear.settings import write_settings

def read_object(name: str, directory: str, env: Dict[str, Any] = None) -> Any:
    """
    env: optional dict where the loaded object is stored
    """
    fname = f"{directory}/{name}.pkl"

    if not os.path.exists(fname):
        raise FileNotFoundError(f"File {fname} does not exist!")
    with open(fname, "rb") as f:
        obj = pickle.load(f)

    if env is not None:
        env[name] = obj
    return obj

def save_object(obj: Any, directory: str, name: str) -> None:
    if name is None:
        raise ValueError("Must provide an object name")
    fname = f"{directory}/{name}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def write_samples(logfile: str, samples_df: pd.DataFrame) -> None:
    with open(logfile, "a") as f:
        f.write("#" * 80 + "\n")
        f.write("# Samples\n")
        f.write("#" * 80 + "\n")

    samples_df.to_csv(logfile, mode="a", index=False)

    with open(logfile, "a") as f:
        f.write("#" * 80 + "\n")

def saverun(settings, object_list=None):
    outpath = settings["outpath"]

    # Ensure directory exists
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    write_settings(settings)

    if object_list is not None:
        for name, obj in object_list.items():
            save_object(obj, outpath, name=name)
        if "samples" in object_list:
            samples = object_list["samples"]
            write_samples(settings["logfile"], samples)
            samples["Entity"].to_csv(
                f"{outpath}/statnames.txt",
                index=False,
                header=False
            )




