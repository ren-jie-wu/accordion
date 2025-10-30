import json
import joblib
import numpy as np
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, fbeta_score,
    make_scorer, precision_recall_curve, auc
)

def train_xgboost(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: str,
    *,
    feature_sets: dict[str, list[str]] | None = None,
    beta: float = 0.1,
    random_state: int = 42,
    n_search_iter: int = 60,
    search_n_jobs: int = -1,
    use_scaler: bool = False
):
    """
    Run the full SIMBA+ model training pipeline end-to-end, with support for
    custom feature combinations.

    Args:
        df (pd.DataFrame): Input data containing features and 'label'
        dataset_name (str): Prefix name for saved outputs (e.g., "crispr")
        output_dir (str): Directory to store model outputs
        feature_sets (dict[str, list[str]], optional):
            Custom mapping of model names → feature column names.
            If None, all numeric columns (except label) will be used as a single feature set.
        beta, random_state, n_search_iter, search_n_jobs, use_scaler: Training settings.

    Returns:
        dict: {
            "results_meta": metadata and model info,
            "metrics_df": per-chromosome metrics,
            "predictions_df": combined predictions
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    # Preprocess dataset into feature subsets (flexible feature sets)
    print(f"Preprocessing dataset '{dataset_name}'...")
    data = preprocess_data(df, dataset_name, feature_sets=feature_sets)

    # Train and save all models
    target_key = f'{dataset_name}_train_y'
    print(f"Starting training for {len(feature_sets)} model(s)...")
    results = train_and_save_models(
        data,
        target_key,
        output_dir,
        beta=beta,
        random_state=random_state,
        n_search_iter=n_search_iter,
        search_n_jobs=search_n_jobs,
        use_scaler=use_scaler
    )

    # Reload all saved outputs for downstream analysis
    print("Reloading saved results for summary...")
    results_obj, metrics_df, predictions_df = load_results(output_dir)

    print(f"\n Finished full pipeline for '{dataset_name}'")
    print(f"  - Models saved to: {output_dir}")
    print(f"  - Metrics shape: {metrics_df.shape}")
    print(f"  - Predictions shape: {predictions_df.shape}")

    return {
        "results_meta": results_obj,
        "metrics_df": metrics_df,
        "predictions_df": predictions_df,
    }


def preprocess_data(df, name, feature_sets=None):
    """
    Preprocess the input dataframe for training.

    Args:
        df (pd.DataFrame): Input with features and label
        name (str): Dataset name prefix
        feature_sets (dict[str, list[str]], optional):
            Custom feature group definitions.
            Example:
                {
                    "simba_only": ["SIMBA_score"],
                    "simba_distance": ["SIMBA_score", "1/Distance"],
                    "custom_combo": ["SIMBA_plus_path_score", "GC_content", "Enhancer_H3K27ac"]
                }

    Returns:
        dict[str, pd.DataFrame]: Feature subsets ready for model training.
    """
    # Default: all numeric except label/id
    if "label" not in df.columns:
        raise ValueError("Input dataframe must contain 'label' column.")
    if feature_sets is None:
        raise ValueError("feature_sets must be provided for preprocessing and model training.")

    out = {}
    for model_name, cols in feature_sets.items():
        # Ensure all listed features exist in df
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            print(f"Skipping '{model_name}' — no valid columns found.")
            continue
        out[model_name] = df[valid_cols + ["peak_gene_pair"]]

    # Always include label table
    out[f"{name}_train_y"] = df[["label", "peak_gene_pair"]]
    return out

# =========================================================
# 2️⃣ Metric Utilities (AUERC & Enrichment-Recall)
# =========================================================

def extract_chromosome(peak_gene_pair: str) -> str:
    """Extract chromosome prefix from a peak_gene_pair string."""
    s = str(peak_gene_pair)
    return s.split('_')[0] if '_' in s else s


def enrichment_recall(df, col, min_recall=0.0, max_recall=1.0,
                      full_info=False, extrapolate=False):
    """
    Compute enrichment-recall curve.

    - Sorts pairs by prediction score (descending)
    - Computes cumulative enrichment among linked (pred>0) pairs
    - Optionally extrapolates tail of the curve

    Returns:
        DataFrame of recall and enrichment values
    """
    tmp = df[[col, "gold"]].copy()
    tmp = tmp.sort_values(by=col, ascending=False).reset_index(drop=True)

    tmp["linked"] = (tmp[col] > 0).astype(int)
    tmp["linked_cum"] = tmp["linked"].cumsum()
    tmp["gold_cum"] = tmp["gold"].cumsum()

    max_gold = tmp["gold_cum"].max()
    if max_gold == 0:
        return pd.DataFrame(columns=["recall", "enrichment"])

    tmp["recall"] = tmp["gold_cum"] / max_gold
    enrich_denom = max_gold / len(tmp)
    tmp["enrichment"] = (tmp["gold_cum"] / np.maximum(tmp["linked_cum"], 1)) / max(enrich_denom, 1e-12)
    tmp = tmp[tmp[col] > 0][["recall", "enrichment"]]

    if extrapolate and not tmp.empty:
        last_point = tmp.iloc[-1]
        last_recall, last_enrichment = last_point["recall"], last_point["enrichment"]
        num_new_points = int((df[col] == 0).sum() * 0.1)
        if num_new_points > 0:
            recall_increment = (1 - last_recall) / num_new_points
            enrichment_increment = (last_enrichment - 1) / num_new_points
            new_recall = [last_recall + recall_increment * i for i in range(1, num_new_points + 1)]
            new_enrichment = [last_enrichment - enrichment_increment * i for i in range(1, num_new_points + 1)]
            extrapolated_er = pd.DataFrame({"recall": new_recall, "enrichment": new_enrichment})
            tmp = pd.concat([tmp, extrapolated_er], ignore_index=True)

    return tmp if full_info else tmp.drop_duplicates(subset=["recall"], keep="first")


def auerc_old(y_true, y_pred, min_recall=0.0, max_recall=1.0, extrapolate=False):
    """Compute AUERC (average enrichment across recall)."""
    df = pd.DataFrame({"pred": y_pred, "gold": y_true})
    er = enrichment_recall(df, "pred", min_recall, max_recall, extrapolate=extrapolate)
    return float(np.average(er["enrichment"])) if not er.empty else 0.0


def _select_numeric_features(df, drop_cols=("chromosome", "peak_gene_pair", "label")):
    """Return numeric-only subset of features."""
    cols = [c for c in df.columns if c not in drop_cols]
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return df[num_cols], num_cols


# =========================================================
# 3️⃣ XGBoost Training (Leave-One-Chromosome-Out CV)
# =========================================================

def train_xgboost_model_leave_one_chr_out(
    train_x, train_y, *,
    beta=0.1,
    random_state=1,
    n_search_iter=60,
    search_n_jobs=-1,
    use_scaler=False
):
    """
    Leave-One-Chromosome-Out (LOCO) training with RandomizedSearchCV.

    Performs:
      - Per-chromosome holdout validation
      - Stratified inner CV hyperparameter search
      - AUERC-based scoring

    Returns:
        (fold_models, results_df_allchrom, prediction_results, metrics_summary, feature_names)
    """
    # Merge features and labels
    if "peak_gene_pair" not in train_x.columns:
        train_x = train_x.reset_index()
    if "peak_gene_pair" not in train_y.columns:
        train_y = train_y.reset_index()

    ty = train_y[["peak_gene_pair", "label"]].drop_duplicates("peak_gene_pair")
    tx = train_x.copy()

    tx["chromosome"] = tx["peak_gene_pair"].apply(extract_chromosome)
    ty["chromosome"] = ty["peak_gene_pair"].apply(extract_chromosome)

    data = pd.merge(tx, ty, on=["peak_gene_pair", "chromosome"], how="inner", validate="1:1")
    if data.empty:
        raise ValueError("After merging, no rows remain — check inputs.")
    data["label"] = data["label"].astype(np.int8)

    # Only use chromosomes with positives
    chrom_has_pos = data.groupby("chromosome")["label"].sum()
    chromosomes = [c for c, s in chrom_has_pos.items() if s > 0]

    # Class balance
    pos, neg = (data["label"] == 1).sum(), (data["label"] == 0).sum()
    ratio = neg / max(pos, 1)

    # Features
    feature_df, feature_names = _select_numeric_features(data)

    # Scoring function
    auerc_scorer = make_scorer(auerc_old, greater_is_better=True, needs_proba=True)

    # Parameter grid
    param_grid = {
        "max_depth": [2, 3, 4, 5, 6],
        "min_child_weight": [1, 3, 5, 10],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "n_estimators": [3000, 5000, 6000, 9000],
        "gamma": [0, 0.5, 1, 1.5, 2],
        "reg_lambda": [1, 3, 10, 30],
        "reg_alpha": [0, 0.5, 1, 3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "scale_pos_weight": [ratio],
        "n_jobs": [1],
        "random_state": [random_state],
    }

    results = []
    all_peak_gene_ids, all_y_true, all_y_pred_prob, all_chroms = [], [], [], []
    fold_models = []

    for chromosome in chromosomes:
        print(f"[LOCO] Holding out {chromosome} for validation")

        is_val = data["chromosome"] == chromosome
        train_df = data.loc[~is_val].reset_index(drop=True)
        val_df = data.loc[is_val].reset_index(drop=True)

        X_train, y_train = train_df[feature_names], train_df["label"].values
        X_val, y_val = val_df[feature_names], val_df["label"].values

        # Optional scaling
        scaler = None
        if use_scaler:
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        base = xgb.XGBClassifier(objective="binary:logistic", eval_metric="aucpr")

        random_search = RandomizedSearchCV(
            base,
            param_distributions=param_grid,
            scoring=auerc_scorer,
            n_iter=n_search_iter,
            cv=cv,
            n_jobs=search_n_jobs,
            verbose=0,
            random_state=random_state,
            error_score="raise"
        )

        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_
        print(f"  ↳ Best params: {best_params}")

        clf = xgb.XGBClassifier(
            **best_params,
            objective="binary:logistic",
            eval_metric="aucpr",
            early_stopping_rounds=100,
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        fold_models.append({"model": clf, "scaler": scaler, "feature_names": feature_names})

        # Validation predictions
        y_val_prob = clf.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_prob >= 0.5).astype(int)

        # Metrics
        p_v, r_v, _ = precision_recall_curve(y_val, y_val_prob)
        val_pr_auc = auc(r_v, p_v)
        val_auerc = auerc_old(y_val, y_val_prob)

        results.append({
            "Heldout Chromosome": chromosome,
            "Val Accuracy": accuracy_score(y_val, y_val_pred),
            "Val Recall": recall_score(y_val, y_val_pred, zero_division=0),
            "Val Precision": precision_score(y_val, y_val_pred, zero_division=0),
            "Val F-beta": fbeta_score(y_val, y_val_pred, beta=beta, zero_division=0),
            "Val AUERC": val_auerc,
            "Val PR-AUC": val_pr_auc,
            "Best Params": best_params,
            "Best Rounds": getattr(clf, "best_iteration", None),
        })

        all_peak_gene_ids.extend(val_df["peak_gene_pair"])
        all_y_true.extend(y_val)
        all_y_pred_prob.extend(y_val_prob)
        all_chroms.extend([chromosome] * len(y_val_prob))

    results_df_allchrom = pd.DataFrame(results)
    prec_all, rec_all, _ = precision_recall_curve(all_y_true, all_y_pred_prob)
    pr_auc_all = auc(rec_all, prec_all)
    auerc_all = auerc_old(all_y_true, all_y_pred_prob)

    prediction_results = pd.DataFrame({
        "peak_gene_pair": all_peak_gene_ids,
        "true_label": all_y_true,
        "predicted_prob": all_y_pred_prob,
        "chromosome": all_chroms
    })

    metrics_summary = {"pr_auc_point": pr_auc_all, "auerc_point": auerc_all}
    return fold_models, results_df_allchrom, prediction_results, metrics_summary, feature_names


# =========================================================
# 4️⃣ Model Saving / Loading
# =========================================================

def train_and_save_models(
    data, target_key, output_dir,
    *, beta=0.1, random_state=42,
    n_search_iter=60, search_n_jobs=-1, use_scaler=False
):
    """
    Train and save models using train_xgboost_model_leave_one_chr_out().
    Saves all model bundles, predictions, and metadata under output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for feature_key in data.keys():
        if "train_y" in feature_key:
            continue  # Skip target key
        print(f"\n=== Training model: {feature_key} ===")
        fold_models, results_df, prediction_results, metrics_summary, feature_names = \
            train_xgboost_model_leave_one_chr_out(
                data[feature_key], data[target_key],
                beta=beta,
                random_state=random_state,
                n_search_iter=n_search_iter,
                search_n_jobs=search_n_jobs,
                use_scaler=use_scaler
            )

        # Save model folds
        model_paths = []
        for i, bundle in enumerate(fold_models):
            path = os.path.join(output_dir, f"{feature_key}_fold{i}.pkl")
            joblib.dump(bundle, path)
            model_paths.append(path)

        # Save predictions and metrics
        preds_path = os.path.join(output_dir, f"{feature_key}_predictions.csv")
        prediction_results.to_csv(preds_path, index=False)

        per_chr_path = os.path.join(output_dir, f"{feature_key}_per_chromosome_metrics.csv")
        results_df.to_csv(per_chr_path, index=False)

        # Metadata
        meta = {
            "model_name": feature_key,
            "feature_key": feature_key,
            "feature_names": feature_names,
            "fold_model_paths": model_paths,
            "metrics": metrics_summary,
            "settings": {
                "beta": beta,
                "random_state": random_state,
                "n_search_iter": n_search_iter,
                "use_scaler": use_scaler
            }
        }
        meta_path = os.path.join(output_dir, f"{feature_key}_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        results[feature_key] = {
            "fold_model_paths": model_paths,
            "predictions_path": preds_path,
            "per_chromosome_metrics_path": per_chr_path,
            "metadata_path": meta_path,
            **metrics_summary
        }

        print(f"Saved {len(model_paths)} folds → {output_dir}")

    results_path = os.path.join(output_dir, "results.pkl")
    joblib.dump(results, results_path)
    print(f"\nSaved summary → {results_path}")

    return results


def load_results(output_dir: str):
    """
    Reload results produced by train_and_save_models() into memory.
    Combines all predictions and per-chromosome metrics.
    """
    results_path = os.path.join(output_dir, "results.pkl")
    results = joblib.load(results_path)

    all_results_list = []
    all_predictions = None

    for model_name, meta in results.items():
        # Per-chromosome metrics
        df_chr = pd.read_csv(meta["per_chromosome_metrics_path"])
        df_chr["model"] = model_name
        all_results_list.append(df_chr)

        # Predictions
        df_pred = pd.read_csv(meta["predictions_path"])
        df_pred = df_pred.rename(columns={"predicted_prob": f"pred_prob_{model_name}"})
        if all_predictions is None:
            all_predictions = df_pred
        else:
            all_predictions = pd.merge(
                all_predictions, df_pred,
                on=["peak_gene_pair", "true_label", "chromosome"], how="outer"
            )

    return results, pd.concat(all_results_list, ignore_index=True), all_predictions