import pandas as pd

def read_CRToolSamples(inputfile: str) -> pd.DataFrame:
    """Read samples from input file
    
    Parameters:
        inputfile (str): Path to the input CSV file containing samples
    Returns:
        samples (pd.DataFrame): DataFrame containing the samples data
    """

    print("INFO: Reading samples... ")
    samples = pd.read_csv(inputfile, sep=',', header=0)

    expected_colnames = ["Entity", "Metric", "Date", "Value", "Uncertainty", "MDC Value"]

    if samples.shape[1] != 6:
        raise ValueError(f"ERROR, samples has {samples.shape[1]} columns instead of 6. Check file '{inputfile}'!")

    if list(samples.columns) != expected_colnames:
        print(f"INFO: renaming colnames of samples to {expected_colnames}. To avoid this message, change the header of the inputfile {inputfile}")
        samples.columns = expected_colnames

    samples['Date'] = samples['Date'].astype(str).str.strip()
    samples['Date'] = pd.to_datetime(samples['Date'], format="%Y-%m-%d %H:%M")

    print("done\n")
    return samples