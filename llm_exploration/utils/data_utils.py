import json
import os
from typing import Optional, List, Any


def read_json(fname: str) -> Any:
    """
    Given a filename, reads a json file and returns the data stored inside.

    Input:
        fname (str):
            Name of the file to be read.

    Output:
        data (Any):
            The data loaded from the json file.
    """

    assert os.path.isfile(fname)
    assert fname.endswith(".json")

    with open(fname, "r") as file:
        data = json.load(file)

    return data


def write_json(
    data: Any,
    fname: str,
) -> None:
    """
    Given a data and the filename, writes the data to the specified
    fname.
    If the directory that the specified filename should be in
    does not exist, then it creates the directory first.

    Input:
        data (Any):
            the data that needs to stored in a json format.

        fname (str):
            path to the file where the data needs to be saved.

    Output:
        None
    """

    assert isinstance(fname, str) and fname.endswith(".json")
    splits = fname.split("/")[:-1]
    root_dir = "/".join(splits)
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir, exist_ok=True)

    with open(fname, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def find_all_files(
    root_dir: str,
    file_suffix: Optional[str] = None,
) -> List[str]:
    """
    Given the root directory, returns the absolute path to all files.
    Filters by suffix if given one.

    Input:
        root_dir (str): The absolute path to the root directory where all files should be in.
        file_suffix (Optional[str]): The suffix (e.g., json) to filter by.
            Default: None

    Output:
        A List of str, with absolute paths to all files in the directory with the given suffix.

    Example Usage:
        all_data_files = find_all_files(
            root_dir='/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_chunked',
            file_suffix='json',
        )

        for data_file in all_data_files:
            pass
    """
    if not os.path.isdir(root_dir):
        raise ValueError(f"Given directory {root_dir} is not a directory.")

    all_sub_file_or_dirs = os.listdir(root_dir)

    all_files = []
    for sub_file_or_dir in all_sub_file_or_dirs:
        absolute_path = os.path.join(root_dir, sub_file_or_dir)
        if os.path.isfile(absolute_path):
            if file_suffix is None or absolute_path.endswith(file_suffix):
                all_files.append(absolute_path)

    return all_files
