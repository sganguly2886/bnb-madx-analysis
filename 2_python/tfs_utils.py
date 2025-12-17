import pandas as pd

def read_tfs(path):
    """
    Read a MAD-X TFS file into a pandas DataFrame.
    """
    data_lines = []
    colnames = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("@"):
                continue
            if line.startswith("*"):
                colnames = line.split()[1:]
                continue
            if line.startswith("$"):
                continue
            data_lines.append(line.split())

    if colnames is None:
        raise RuntimeError(f"No column header found in {path}")

    df = pd.DataFrame(data_lines, columns=colnames)

    # convert numeric columns
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    return df
