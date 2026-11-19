import argparse

from src.data_work import load_data, split_data
from src.model_work import build_pipeline
from src.quality_work import evaluate, tune_k


def main() -> None:
    parser = argparse.ArgumentParser(description="k-NN classification of schizophrenia proneness")
    parser.add_argument("--data", default="data/SchizophreniaSymptomnsData.csv", help="path to the Kaggle CSV")
    parser.add_argument("--k", type=int, default=3, help="number of neighbors")
    parser.add_argument("--tune", action="store_true", help="also cross-validate k on the training set")
    parser.add_argument("--keep-marital-status", action="store_true", help="do not drop Marital_Status")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    df = load_data(args.data, drop_marital_status=not args.keep_marital_status)
    x_train, x_test, y_train, y_test = split_data(df)

    if args.tune:
        table = tune_k(lambda k: build_pipeline(x_train, n_neighbors=k), x_train, y_train, args.out)
        print(table.round(4).to_string(index=False))

    model = build_pipeline(x_train, n_neighbors=args.k).fit(x_train, y_train)
    metrics = evaluate(model, x_train, y_train, x_test, y_test, args.out)

    print(f"k = {args.k}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
