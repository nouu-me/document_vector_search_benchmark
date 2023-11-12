import pandas as pd
from tabulate import tabulate


def main(input_csv_file: str) -> None:
    with open(input_csv_file) as fin:
        df = pd.read_csv(fin)
    df = df.sort_values(["dataset", "Recall@1"], ascending=[True, False]).reset_index(drop=True)
    print(tabulate(df, headers=df.columns, tablefmt="github"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv_file", type=str)
    args = parser.parse_args()
    main(args.input_csv_file)
