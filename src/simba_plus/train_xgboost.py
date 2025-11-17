
import pandas as pd
from simba_plus.discovery import build_evalset, model_training, plot_utils
from simba_plus.discovery import candidate_links as pgl
import pyranges as pr
import os
from simba_plus.discovery import add_features as score
from simba_plus.discovery.model_training import load_results
import argparse
import ast

def train_supervised(eval_df, feature_sets, dataset_name, output_path):
    results_obj, metrics_df, preds_df = model_training.train_xgboost(
            df=eval_df,
            dataset_name=dataset_name,
            output_dir=output_path,
            feature_sets=feature_sets,
            n_search_iter=30,
            search_n_jobs=8,
            random_state=1
        )
    return results_obj, metrics_df, preds_df

def build_feature_sets(feature_sets: dict, output_path: str, python_list_format: bool = False):    
    rows = []
    for name, features in feature_sets.items():
        if python_list_format:
            features_str = str(features)                    # "['a','b']"
        else:
            features_str = ",".join(features)               # "a,b"

        rows.append({
            "feature_set_name": name,
            "features": features_str
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Feature sets CSV written to: {output_path}")
    return df


# ======================================================
# CLI SUPPORT
# ======================================================

import pandas as pd
import ast

def parse_feature_sets(csv_path):
    df = pd.read_csv(csv_path)

    if "feature_set_name" not in df.columns or "features" not in df.columns:
        raise ValueError("CSV must contain columns: 'feature_set_name' and 'features'")

    feature_sets = {}
    for _, row in df.iterrows():
        name = row["feature_set_name"]
        value = str(row["features"]).strip()

        # --- Case 1: Python list string ---
        # Examples: "['a','b']", ["x", "y"]
        if value.startswith("[") and value.endswith("]"):
            try:
                features = ast.literal_eval(value)
                # ensure list of strings
                features = [str(f).strip() for f in features]
            except Exception:
                raise ValueError(f"Invalid Python list syntax in features column: {value}")

        # --- Case 2: comma-separated list ---
        else:
            features = [f.strip() for f in value.split(",") if f.strip()]

        feature_sets[name] = features

    return feature_sets

def add_parser(subparser):
    subparser.add_argument("--eval_csv", required=True, help="Evaluation dataframe CSV.")
    subparser.add_argument("--feature_sets_csv", required=True, help="Feature sets CSV.")
    subparser.add_argument("--dataset_name", required=True, help="Dataset identifier.")
    subparser.add_argument("--output_path", required=True, help="Directory for outputs.")
    return subparser

def main(args):
    eval_df = pd.read_csv(args.eval_csv)
    feature_sets = parse_feature_sets(args.feature_sets_csv)

    os.makedirs(args.output_path, exist_ok=True)

    _, metrics_df, preds_df = train_supervised(
        eval_df=eval_df,
        feature_sets=feature_sets,
        dataset_name=args.dataset_name,
        output_path=args.output_path,
    )

    metrics_df.to_csv(os.path.join(args.output_path, "metrics.csv"), index=False)
    preds_df.to_csv(os.path.join(args.output_path, "preds.csv"), index=False)

    print(f"Finished supervised training → {args.output_path}")
