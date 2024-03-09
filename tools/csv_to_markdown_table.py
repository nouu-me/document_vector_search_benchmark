import pandas as pd
from tabulate import tabulate


def main(input_csv_file: str) -> None:
    df = pd.read_csv(input_csv_file)
    # Pivot the DataFrame to match the desired format
    df_pivot = df.pivot_table(index=["embedding"],
                              columns=["dataset"],
                              values=["Recall@1", "Recall@3", "Recall@5", "Recall@10", "Recall@100"],
                              aggfunc='first')
    # Flatten the MultiIndex columns and join levels with underscore
    df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]
    # Reorder columns to group by dataset rather than metric
    datasets = sorted(set(col.split('_')[1] for col in df_pivot.columns))
    metrics = ["Recall@1", "Recall@3", "Recall@5", "Recall@10", "Recall@100"]
    ordered_columns = [f"{metric}_{dataset}" for dataset in datasets for metric in metrics]
    df_pivot = df_pivot[ordered_columns]
    # Reset index to get 'embedding' back as a column
    df_pivot.reset_index(inplace=True)
    # Rename 'embedding' column to 'Model'
    df_pivot.rename(columns={'embedding': 'Model'}, inplace=True)
    # Sort by Recall@3 for JSQuAD-v1.1-valid
    sort_column = next(col for col in ordered_columns if "Recall@3_JSQuAD-v1.1-valid" in col)
    
    df_pivot.sort_values(by=sort_column, inplace=True, ascending=False)
    # Adjust column names to remove underscore
    df_pivot.columns = [col.replace('_', ' ') for col in df_pivot.columns]
    try:
        # Drop the less informative columns if they're present to help readability
        df_pivot.drop(columns=["Recall@100 JSQuAD-v1.1-valid"], inplace=True)
        df_pivot.drop(columns=["Recall@1 MIRACL-v1.0-dev"], inplace=True)
    except:
        pass
    # Reset index after sorting to reflect the new order
    df_pivot.reset_index(drop=True, inplace=True)
    print(tabulate(df_pivot, headers='keys', tablefmt="github"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv_file", type=str)
    args = parser.parse_args()
    main(args.input_csv_file)
