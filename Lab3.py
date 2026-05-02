"""
Lab 3 - ANOVA
Dataset: WorldEnergy.csv

"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except ImportError as exc:
    raise ImportError(
        "This program requires statsmodels. Install it with: pip install statsmodels"
    ) from exc


warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------
# 1. User settings
# -----------------------------
SELECTED_COUNTRIES = [
    "Malaysia",
    "Indonesia",
    "Thailand",
    "Vietnam",
    "Singapore",
    "Philippines",
]

START_YEAR = 1985
END_YEAR = 2024
VALUE_COLUMN = "renewables_share_elec"


# -----------------------------
# 2. Helper functions
# -----------------------------
def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the WorldEnergy.csv dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Cannot find '{csv_path}'. Put WorldEnergy.csv in this folder or use --csv to set the path."
        )

    df = pd.read_csv(csv_path)
    print("=" * 70)
    print("Dataset loaded successfully")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("=" * 70)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select useful columns, filter countries and years, remove missing values,
    and prepare data for one-way ANOVA.
    """
    required_columns = ["country", "year", VALUE_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required column(s): {missing_columns}")

    data = df[required_columns].copy()
    data = data.rename(
        columns={
            "country": "Country",
            "year": "Year",
            VALUE_COLUMN: "RenewablesShareElec",
        }
    )

    # Convert to numeric. Invalid values become NaN and are removed later.
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    data["RenewablesShareElec"] = pd.to_numeric(data["RenewablesShareElec"], errors="coerce")

    # Filter the analysis sample.
    data = data[
        (data["Country"].isin(SELECTED_COUNTRIES))
        & (data["Year"] >= START_YEAR)
        & (data["Year"] <= END_YEAR)
    ]

    # Remove rows with missing values in the dependent variable.
    data = data.dropna(subset=["Country", "Year", "RenewablesShareElec"])

    # Keep the countries in a consistent order.
    data["Country"] = pd.Categorical(
        data["Country"], categories=SELECTED_COUNTRIES, ordered=True
    )
    data = data.sort_values(["Country", "Year"]).reset_index(drop=True)

    print("Data preprocessing completed")
    print(f"Selected countries: {', '.join(SELECTED_COUNTRIES)}")
    print(f"Selected years: {START_YEAR}-{END_YEAR}")
    print(f"Final rows for ANOVA: {len(data)}")
    print("Observations per country:")
    print(data.groupby("Country", observed=False).size())
    print("=" * 70)

    if data["Country"].nunique() < 2:
        raise ValueError("ANOVA requires at least two groups after preprocessing.")

    if data.groupby("Country", observed=False).size().min() < 2:
        raise ValueError("Each group should contain at least two observations for ANOVA.")

    return data


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Generate descriptive statistics for each country."""
    desc = (
        data.groupby("Country", observed=False)["RenewablesShareElec"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    desc["mean"] = desc["mean"].round(3)
    desc["std"] = desc["std"].round(3)
    desc["min"] = desc["min"].round(3)
    desc["median"] = desc["median"].round(3)
    desc["max"] = desc["max"].round(3)

    print("Descriptive statistics")
    print(desc.to_string(index=False))
    print("=" * 70)
    return desc

def run_anova(data: pd.DataFrame):
    """Run one-way ANOVA and calculate effect size."""
    print("Research question")
    print(
        "Is there a statistically significant difference in mean renewable electricity share "
        "among the selected Southeast Asian countries?"
    )
    print("Null hypothesis H0: all selected countries have the same mean renewable electricity share.")
    print("Alternative hypothesis H1: at least one country has a different mean.")
    print("=" * 70)

    model = smf.ols("RenewablesShareElec ~ C(Country)", data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Calculate eta squared: SS_between / SS_total
    ss_between = anova_table.loc["C(Country)", "sum_sq"]
    ss_error = anova_table.loc["Residual", "sum_sq"]
    eta_squared = ss_between / (ss_between + ss_error)
    anova_table["eta_squared"] = [eta_squared, np.nan]

    print("One-way ANOVA table")
    print(anova_table.round(6))
    print(f"Effect size, eta squared: {eta_squared:.4f}")
    print("=" * 70)

    return model, anova_table

def create_visualizations(data: pd.DataFrame, output_dir: str) -> None:
    """Create simple plots for the report."""
    os.makedirs(output_dir, exist_ok=True)

    # Boxplot by country
    plt.figure(figsize=(10, 6))
    data.boxplot(column="RenewablesShareElec", by="Country", grid=False, rot=30)
    plt.title("Renewable Electricity Share by Country")
    plt.suptitle("")
    plt.xlabel("Country")
    plt.ylabel("Renewables share of electricity (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_renewables_by_country_boxplot.png"), dpi=300)
    plt.close()

    # Time trend by country
    plt.figure(figsize=(10, 6))
    for country, group in data.groupby("Country", observed=False):
        plt.plot(group["Year"], group["RenewablesShareElec"], marker="o", label=country)
    plt.title("Trend of Renewable Electricity Share")
    plt.xlabel("Year")
    plt.ylabel("Renewables share of electricity (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_renewables_trend.png"), dpi=300)
    plt.close()

    # Bar chart of mean with standard error
    summary = (
        data.groupby("Country", observed=False)["RenewablesShareElec"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])

    plt.figure(figsize=(10, 6))
    plt.bar(summary["Country"].astype(str), summary["mean"], yerr=summary["se"], capsize=5)
    plt.title("Mean Renewable Electricity Share with Standard Error")
    plt.xlabel("Country")
    plt.ylabel("Mean renewables share of electricity (%)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_group_means.png"), dpi=300)
    plt.close()

    print(f"Figures saved in: {output_dir}")
    print("=" * 70)


def save_outputs(
    data: pd.DataFrame,
    desc: pd.DataFrame,
    anova_table: pd.DataFrame,
    output_dir: str,
) -> None:
    """Save all output tables to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    data.to_csv(os.path.join(output_dir, "cleaned_lab3_dataset.csv"), index=False)
    desc.to_csv(os.path.join(output_dir, "descriptive_statistics.csv"), index=False)
    anova_table.to_csv(os.path.join(output_dir, "anova_table.csv"))
    print(f"Output tables saved in: {output_dir}")
    print("=" * 70)


def print_final_guide(anova_table: pd.DataFrame) -> None:
    """Print a short guide for students to write their own report discussion."""
    p_value = anova_table.loc["C(Country)", "PR(>F)"]

    print("1. Explain why this subset is chosen:")
    print("   - The dependent variable is numeric: renewables_share_elec.")
    print("   - The independent variable is categorical: country.")
    print("   - The selected countries have complete observations from 1985 to 2024.")
    print("   - This makes the data suitable for one-way ANOVA.")
    print("2. Discuss the descriptive statistics and figures.")
    print("3. Use the ANOVA to decide whether to reject H0.")
    if p_value < 0.05:
        print("   Current result: p < 0.05, so the ANOVA result is statistically significant.")
    else:
        print("   Current result: p >= 0.05, so the ANOVA result is not statistically significant.")
    print("4. If significant, use the Tukey table to identify which pairs of countries differ.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Lab 3 ANOVA using WorldEnergy.csv")
    parser.add_argument(
        "--csv",
        default="WorldEnergy.csv",
        help="Path to WorldEnergy.csv. Default: WorldEnergy.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Folder for generated outputs. Default: outputs",
    )
    args = parser.parse_args()

    try:
        df = load_dataset(args.csv)
        data = preprocess_data(df)
        desc = descriptive_statistics(data)
        model, anova_table = run_anova(data)
        create_visualizations(data, args.output)
        save_outputs(
            data=data,
            desc=desc,
            anova_table=anova_table,
            output_dir=args.output,
        )
        print_final_guide(anova_table)

    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
