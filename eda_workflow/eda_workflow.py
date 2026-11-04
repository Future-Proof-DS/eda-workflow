import logging
import os
import re
from typing import Optional, TypedDict

import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)
WORKFLOW_NAME = "eda_workflow"
LOG_PATH = os.path.join(os.getcwd(), "logs/")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
ID_WORDS = {"id", "uuid", "guid", "key"}
DATE_WORDS = {"date", "datetime", "timestamp", "year", "month"}
RATE_WORDS = {
    "price", "rate", "ratio", "pct", "percent", "percentage", "per",
    "avg", "average", "mean", "median", "score", "age",
}
SENTINEL_VALUES = {
    "unknown", "error", "missing", "undefined", "none", "null", "nan",
    "n/a", "na", "?", "-", "",
}
MAX_CATEGORIES = 50


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    with open(prompt_path, "r") as f:
        return f.read()


def mask_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace placeholder strings such as "unknown" or "error" with NaN."""
    masked = df.copy()
    for col in masked.select_dtypes(include=["object", "category"]).columns:
        normalized = masked[col].astype(str).str.strip().str.lower()
        masked[col] = masked[col].mask(normalized.isin(SENTINEL_VALUES))
    return masked


def classify_column_roles(df: pd.DataFrame) -> dict:
    """
    Group columns by what they mean rather than by their pandas dtype.

    A dtype can't tell a total from a price or an ID from a quantity, and
    only measures make sense to add up. Roles come from the column name
    first, then from the values.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset, with sentinels already masked.

    Returns
    -------
    dict
        Column names under "id", "date", "measure", "rate", "categorical"
        and "text".
    """
    roles = {
        "id": [], "date": [], "measure": [],
        "rate": [], "categorical": [], "text": [],
    }

    for col in df.columns:
        words = set(re.sub(r"[^a-z0-9]+", " ", str(col).lower()).split())
        values = df[col].dropna()
        is_numeric = (
            pd.api.types.is_numeric_dtype(values)
            and not pd.api.types.is_bool_dtype(values)
        )

        if words & ID_WORDS:
            roles["id"].append(col)
        elif words & DATE_WORDS:
            roles["date"].append(col)
        elif is_numeric and (words & RATE_WORDS or values.between(0, 1).all()):
            roles["rate"].append(col)
        elif not is_numeric and (
            pd.to_datetime(values, errors="coerce", format="mixed").notna().mean() > 0.8
        ):
            roles["date"].append(col)
        elif values.nunique() == len(df) and not pd.api.types.is_float_dtype(values):
            roles["id"].append(col)
        elif is_numeric:
            roles["measure"].append(col)
        elif values.nunique() <= MAX_CATEGORIES:
            roles["categorical"].append(col)
        else:
            roles["text"].append(col)

    return roles


class EDAWorkflow:
    """
    Exploratory Data Analysis workflow that performs consistent, first-pass analysis of datasets.

    Uses a fixed set of predefined analysis tools to produce structured, tabular outputs.
    Operates sequentially and deterministically through baseline EDA steps.

    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.

    Attributes
    ----------
    response : dict or None
        Stores the full response after invoke_workflow() is called.
    """

    def __init__(
        self,
        model=None,
        log=False,
        log_path=None,
        checkpointer: Optional[object] = None
    ):
        self.model = model
        self.log = log
        self.log_path = log_path
        self.checkpointer = checkpointer
        self.response = None
        self._compiled_graph = make_eda_baseline_workflow(
            model=model,
            log=log,
            log_path=log_path,
            checkpointer=checkpointer
        )

    def invoke_workflow(self, filepath: str, **kwargs):
        """
        Run EDA analysis on the provided dataset.

        Parameters
        ----------
        filepath : str
            Path to the dataset file.
        **kwargs
            Additional arguments passed to the underlying graph invoke method.

        Returns
        -------
        None
            Results are stored in self.response and accessed via getter methods.
        """
        df = pd.read_csv(filepath)

        response = self._compiled_graph.invoke({
            "dataframe": df.to_dict(),
            "results": {},
            "observations": {},
            "current_step": "",
            "summary": "",
            "recommendations": [],
        }, **kwargs)

        self.response = response
        return None

    def get_summary(self):
        """Retrieves the analysis summary."""
        if self.response:
            return self.response.get("summary")

    def get_recommendations(self):
        """Retrieves the recommendations."""
        if self.response:
            return self.response.get("recommendations")

    def get_results(self):
        """Retrieves the full analysis results."""
        if self.response:
            return self.response.get("results")

    def get_observations(self):
        """Retrieves all observations from analysis steps."""
        if self.response:
            return self.response.get("observations")


def make_eda_baseline_workflow(
    model=None,
    log=False,
    log_path=None,
    checkpointer: Optional[object] = None
):
    """
    Factory function that creates a compiled LangGraph workflow for baseline EDA.

    Performs automated first-pass analysis with fixed analysis steps.

    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.

    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow ready to process EDA requests.
    """
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    class EDAState(TypedDict):
        dataframe: dict
        results: dict
        observations: dict[str, list[str]]
        current_step: str
        summary: str
        recommendations: list[str]

    def profile_dataset_node(state: EDAState):
        """Generate dataset profile with basic statistics."""
        logger.info("Profiling dataset")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        masked = mask_sentinels(df)
        roles = classify_column_roles(masked)

        profile = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "column_roles": roles,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "numeric_summary": (
                df[numeric_cols].describe().to_dict() if numeric_cols else {}
            ),
            "categorical_summary": {
                col: masked[col].value_counts().head(10).to_dict()
                for col in categorical_cols
                if col not in roles["id"]
            },
        }

        results["profile_dataset"] = profile

        return {
            "current_step": "profile_dataset",
            "results": results,
        }

    def analyze_missingness_node(state: EDAState):
        """Analyze missing values in the dataset."""
        logger.info("Analyzing missingness")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})

        # Sentinels count as missing, so nulls alone would understate the problem.
        masked = mask_sentinels(df)
        is_sentinel = masked.isnull() & df.notnull()
        sentinel_values = {
            col: df.loc[is_sentinel[col], col].value_counts().to_dict()
            for col in df.columns[is_sentinel.any()]
        }

        missing_count = masked.isnull().sum().to_dict()
        missing_pct = (
            (masked.isnull().sum() / len(df) * 100).round(2).to_dict()
        )

        high_missing = {col: pct for col, pct in missing_pct.items() if pct > 20}

        missingness = {
            "total_rows": len(df),
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "sentinel_values": sentinel_values,
            "high_missing_columns": high_missing,
            "complete_rows": int(masked.dropna().shape[0]),
            "complete_rows_pct": (
                round(masked.dropna().shape[0] / len(df) * 100, 2)
                if len(df) > 0 else 0
            ),
        }

        results["analyze_missingness"] = missingness

        return {
            "current_step": "analyze_missingness",
            "results": results,
        }

    def compute_aggregates_node(state: EDAState):
        """Compute group-by aggregates on key columns."""
        logger.info("Computing aggregates")
        df = mask_sentinels(pd.DataFrame.from_dict(state.get("dataframe")))
        results = state.get("results", {})

        # Only measures are safe to add up, so IDs, dates and rates stay out.
        roles = results["profile_dataset"]["column_roles"]
        measure_cols = roles["measure"]
        categorical_cols = roles["categorical"]

        aggregates = {}
        rows_without_category = {}

        for cat_col in categorical_cols[:3]:
            if df[cat_col].nunique() <= 20:
                for num_col in measure_cols[:3]:
                    key = f"{num_col}_by_{cat_col}"
                    agg = df.groupby(cat_col)[num_col].agg(["sum", "mean", "count"])
                    agg["pct_of_total"] = agg["sum"] / df[num_col].sum() * 100
                    aggregates[key] = (
                        agg.sort_values("sum", ascending=False).round(2).to_dict("index")
                    )

                # groupby drops masked rows, so say how much they leave unattributed.
                ungrouped = df[df[cat_col].isnull()]
                rows_without_category[cat_col] = {
                    "rows": len(ungrouped),
                    "pct_of_rows": round(len(ungrouped) / len(df) * 100, 2),
                    "measure_totals": ungrouped[measure_cols].sum().round(2).to_dict(),
                }

        monthly_trend = {}
        date_spikes = {}
        typical_rows_per_day = None
        date_cols = df[roles["date"]].select_dtypes(exclude="number").columns.tolist()

        if date_cols:
            dates = pd.to_datetime(df[date_cols[0]], errors="coerce", format="mixed")

            monthly = df.groupby(dates.dt.to_period("M"))[measure_cols].sum()
            monthly.insert(0, "rows", df.groupby(dates.dt.to_period("M")).size())
            monthly.index = monthly.index.astype(str)
            monthly_trend = monthly.round(2).to_dict("index")

            # One date far above its usual volume tends to be a default or fill value.
            rows_per_day = df.groupby(dates.dt.date).size()
            typical_rows_per_day = float(rows_per_day.median())
            spikes = rows_per_day[rows_per_day > typical_rows_per_day * 5]
            date_spikes = {str(day): int(rows) for day, rows in spikes.items()}

        results["compute_aggregates"] = {
            "aggregates": aggregates,
            "rows_without_category": rows_without_category,
            "monthly_trend": monthly_trend,
            "date_spikes": date_spikes,
            "typical_rows_per_day": typical_rows_per_day,
            "categorical_columns_used": categorical_cols[:3],
            "measure_columns_used": measure_cols[:3],
            "columns_not_aggregated": roles["id"] + roles["date"] + roles["rate"],
        }

        return {
            "current_step": "compute_aggregates",
            "results": results,
        }

    def analyze_relationships_node(state: EDAState):
        """Analyze relationships between variables."""
        logger.info("Analyzing relationships")
        df = mask_sentinels(pd.DataFrame.from_dict(state.get("dataframe")))
        results = state.get("results", {})

        roles = results["profile_dataset"]["column_roles"]
        numeric_cols = roles["measure"] + roles["rate"]

        relationships = {}

        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr().round(3)
            relationships["correlation_matrix"] = corr_matrix.to_dict()

            strong_correlations = []
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr_val = float(corr_matrix.loc[col1, col2])
                    if abs(corr_val) > 0.7:
                        strong_correlations.append({
                            "column_1": col1,
                            "column_2": col2,
                            "correlation": corr_val
                        })
            relationships["strong_correlations"] = strong_correlations
        else:
            relationships["correlation_matrix"] = {}
            relationships["strong_correlations"] = []
            relationships["note"] = "Not enough numeric columns for correlation analysis"

        # A rate times a measure often defines another measure, as in
        # quantity * price = total. Rows that break the formula are bad data.
        derived_columns = []
        for rate_col in roles["rate"]:
            for measure_col in roles["measure"]:
                for total_col in roles["measure"]:
                    if total_col == measure_col:
                        continue
                    rows = df[[rate_col, measure_col, total_col]].dropna()
                    expected = rows[rate_col] * rows[measure_col]
                    matches = (expected - rows[total_col]).abs() < 0.01
                    if matches.mean() >= 0.5:
                        derived_columns.append({
                            "formula": f"{measure_col} * {rate_col} = {total_col}",
                            "matching_rows_pct": round(float(matches.mean()) * 100, 2),
                            "mismatched_rows": int((~matches).sum()),
                        })
        relationships["derived_columns"] = derived_columns

        # Share of each numeric column's variance explained by the category (0 to 1).
        category_strength = {}
        for cat_col in roles["categorical"][:3]:
            for num_col in numeric_cols:
                rows = df[[cat_col, num_col]].dropna()
                overall_mean = rows[num_col].mean()
                grouped = rows.groupby(cat_col)[num_col]
                between_groups = (
                    grouped.size() * (grouped.mean() - overall_mean) ** 2
                ).sum()
                total_variance = ((rows[num_col] - overall_mean) ** 2).sum()
                if total_variance > 0:
                    category_strength[f"{cat_col}_vs_{num_col}"] = round(
                        float(between_groups / total_variance), 3
                    )
        relationships["category_strength"] = category_strength

        results["analyze_relationships"] = relationships

        return {
            "current_step": "analyze_relationships",
            "results": results,
        }

    def extract_observations_node(state: EDAState):
        """Extract observations from the latest analysis results using LLM."""
        logger.info("Extracting observations")

        current_step = state.get("current_step", "")
        results = state.get("results", {})
        observations = state.get("observations", {})

        if model is None or not current_step or current_step not in results:
            return {"observations": observations}

        step_results = results.get(current_step, {})

        class ObservationOutput(BaseModel):
            observations: list[str] = Field(description="1-2 concise, actionable observations")

        observation_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("extract_observations_system.txt")),
            ("human", load_prompt("extract_observations_human.txt")),
        ])

        chain = observation_prompt | model.with_structured_output(ObservationOutput)
        response = chain.invoke({
            "step_name": current_step.replace("_", " ").title(),
            "results": str(step_results)
        })

        observations[current_step] = response.observations

        return {
            "observations": observations,
        }

    def synthesize_findings_node(state: EDAState):
        """Synthesize accumulated findings into summary and recommendations."""
        logger.info("Synthesizing findings")

        observations = state.get("observations", {})

        if model is None:
            return {
                "summary": "No LLM provided for synthesis",
                "recommendations": [],
            }

        class SynthesisOutput(BaseModel):
            summary: str = Field(description="A concise 2-3 sentence summary of key findings")
            recommendations: list[str] = Field(description="3-5 actionable recommendations")

        all_observations = []
        for step_name, step_obs in observations.items():
            all_observations.append(f"\n{step_name.replace('_', ' ').title()}:")
            for obs in step_obs:
                all_observations.append(f"  - {obs}")

        observations_text = "\n".join(all_observations)

        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("synthesize_findings_system.txt")),
            ("human", load_prompt("synthesize_findings_human.txt")),
        ])

        chain = synthesis_prompt | model.with_structured_output(SynthesisOutput)
        response = chain.invoke({"observations": observations_text})

        return {
            "summary": response.summary,
            "recommendations": response.recommendations,
        }

    workflow = StateGraph(EDAState)

    workflow.add_node("profile_dataset", profile_dataset_node)
    workflow.add_node("extract_observations_1", extract_observations_node)
    workflow.add_node("analyze_missingness", analyze_missingness_node)
    workflow.add_node("extract_observations_2", extract_observations_node)
    workflow.add_node("compute_aggregates", compute_aggregates_node)
    workflow.add_node("extract_observations_3", extract_observations_node)
    workflow.add_node("analyze_relationships", analyze_relationships_node)
    workflow.add_node("extract_observations_4", extract_observations_node)
    workflow.add_node("synthesize_findings", synthesize_findings_node)

    workflow.set_entry_point("profile_dataset")

    workflow.add_edge("profile_dataset", "extract_observations_1")
    workflow.add_edge("extract_observations_1", "analyze_missingness")
    workflow.add_edge("analyze_missingness", "extract_observations_2")
    workflow.add_edge("extract_observations_2", "compute_aggregates")
    workflow.add_edge("compute_aggregates", "extract_observations_3")
    workflow.add_edge("extract_observations_3", "analyze_relationships")
    workflow.add_edge("analyze_relationships", "extract_observations_4")
    workflow.add_edge("extract_observations_4", "synthesize_findings")
    workflow.add_edge("synthesize_findings", END)

    app = workflow.compile(checkpointer=checkpointer, name=WORKFLOW_NAME)

    return app
