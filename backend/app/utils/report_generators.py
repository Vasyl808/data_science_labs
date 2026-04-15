import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Type

import pandas as pd

from app.ml.pipeline import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def _eval_result_to_html(eval_result) -> str:
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        eval_result.save_html(str(tmp_path))
        return tmp_path.read_text(encoding="utf-8")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


class BaseReportGenerator(ABC):
    @abstractmethod
    def generate(self, reference: pd.DataFrame, current: pd.DataFrame) -> str:
        raise NotImplementedError

    @staticmethod
    def _error_html(title: str, message: str) -> str:
        return f"<html><body><h3>{title}</h3><p>{message}</p></body></html>"


class DataDriftReportGenerator(BaseReportGenerator):
    def generate(self, reference: pd.DataFrame, current: pd.DataFrame) -> str:
        from evidently import Dataset, DataDefinition, Report
        from evidently.presets import DataDriftPreset, DataSummaryPreset

        feature_cols = [c for c in FEATURE_COLUMNS if c in current.columns]
        if not feature_cols:
            return self._error_html(
                "Data Drift report unavailable",
                "No matching feature columns found in current data.",
            )

        current_ds = Dataset.from_pandas(
            current[feature_cols],
            data_definition=DataDefinition(),
        )

        if reference is None or reference.empty:
            report = Report([DataSummaryPreset()])
            return _eval_result_to_html(report.run(current_ds, None))

        ref_cols = [c for c in feature_cols if c in reference.columns]
        if not ref_cols:
            return self._error_html(
                "Data Drift report unavailable",
                "No overlapping feature columns between reference and current data.",
            )

        reference_ds = Dataset.from_pandas(
            reference[ref_cols],
            data_definition=DataDefinition(),
        )
        report = Report([DataDriftPreset()])
        return _eval_result_to_html(report.run(current_ds, reference_ds))


class TargetDriftReportGenerator(BaseReportGenerator):
    def _prepare_target(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        if label_col not in df.columns:
            return pd.DataFrame(columns=["target"])

        target = (
            df[[label_col]]
            .rename(columns={label_col: "target"})
            .dropna(subset=["target"])
            .copy()
        )

        if not target.empty and target["target"].dtype == "object":
            target["target"] = target["target"].astype(str).str.strip()

        return target

    def generate(self, reference: pd.DataFrame, current: pd.DataFrame) -> str:
        from evidently import Dataset, DataDefinition, Report
        from evidently.metrics import ValueDrift
        from evidently.presets import DataSummaryPreset

        current_target = self._prepare_target(current, label_col="true_label")
        if current_target.empty:
            return self._error_html(
                "Target Drift report unavailable",
                "No valid true_label values found in current data.",
            )

        current_ds = Dataset.from_pandas(
            current_target,
            data_definition=DataDefinition(categorical_columns=["target"]),
        )

        if reference is None or reference.empty:
            report = Report([DataSummaryPreset()])
            return _eval_result_to_html(report.run(current_ds, None))

        reference_target = self._prepare_target(reference, label_col="target")
        if reference_target.empty:
            report = Report([DataSummaryPreset()])
            return _eval_result_to_html(report.run(current_ds, None))

        reference_ds = Dataset.from_pandas(
            reference_target,
            data_definition=DataDefinition(categorical_columns=["target"]),
        )

        report = Report([ValueDrift(column="target")])
        return _eval_result_to_html(report.run(current_ds, reference_ds))


class ClassificationReportGenerator(BaseReportGenerator):
    @staticmethod
    def _safe_to_int(series: pd.Series, name: str) -> pd.Series:
        try:
            return series.astype(int)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Cannot cast column '{name}' to int. "
                f"Unique values sample: {series.dropna().unique()[:5].tolist()}"
            ) from exc

    def generate(self, reference: pd.DataFrame, current: pd.DataFrame) -> str:
        from evidently import Dataset, DataDefinition, Report
        from evidently import BinaryClassification
        from evidently.presets import ClassificationPreset, DataSummaryPreset

        current_labeled = current[current["true_label"].notna()].copy()
        if current_labeled.empty:
            return self._error_html(
                "Classification report unavailable",
                "No labeled data found. Update predictions with true_label first.",
            )

        feature_cols = [c for c in FEATURE_COLUMNS if c in reference.columns and c in current_labeled.columns]

        ref = reference[feature_cols + ["target", "prediction"]].copy() if reference is not None and not reference.empty else pd.DataFrame()
        cur = current_labeled[feature_cols + ["true_label", "prediction"]].copy()

        cur["target"] = self._safe_to_int(cur["true_label"], "current.true_label")
        cur["prediction"] = self._safe_to_int(cur["prediction"], "current.prediction")

        if ref.empty:
            current_ds = Dataset.from_pandas(
                cur[["target", "prediction"]],
                data_definition=DataDefinition(
                    classification=[BinaryClassification(target="target", prediction_labels="prediction")],
                    categorical_columns=["target", "prediction"],
                ),
            )
            report = Report([DataSummaryPreset()])
            return _eval_result_to_html(report.run(current_ds, None))

        ref["target"] = self._safe_to_int(ref["target"], "reference.target")
        ref["prediction"] = self._safe_to_int(ref["prediction"], "reference.prediction")

        definition = DataDefinition(
            classification=[BinaryClassification(target="target", prediction_labels="prediction")],
            categorical_columns=["target", "prediction"],
        )

        ref_ds = Dataset.from_pandas(ref, data_definition=definition)
        cur_ds = Dataset.from_pandas(cur[["target", "prediction"]], data_definition=definition)

        report = Report([ClassificationPreset()])
        return _eval_result_to_html(report.run(cur_ds, ref_ds))


class DataQualityReportGenerator(BaseReportGenerator):
    def generate(self, reference: pd.DataFrame, current: pd.DataFrame) -> str:
        from evidently import Dataset, DataDefinition, Report
        from evidently.presets import DataSummaryPreset

        feature_cols = [c for c in FEATURE_COLUMNS if c in current.columns]
        if not feature_cols:
            return self._error_html(
                "Data Quality report unavailable",
                "No matching feature columns found in current data.",
            )

        current_ds = Dataset.from_pandas(
            current[feature_cols],
            data_definition=DataDefinition(),
        )

        if reference is not None and not reference.empty:
            ref_cols = [c for c in feature_cols if c in reference.columns]
            if ref_cols:
                reference_ds = Dataset.from_pandas(
                    reference[ref_cols],
                    data_definition=DataDefinition(),
                )
                report = Report([DataSummaryPreset()])
                return _eval_result_to_html(report.run(current_ds, reference_ds))

        report = Report([DataSummaryPreset()])
        return _eval_result_to_html(report.run(current_ds, None))


class ReportGeneratorFactory:
    _generators: Dict[str, Type[BaseReportGenerator]] = {
        "data_drift": DataDriftReportGenerator,
        "target_drift": TargetDriftReportGenerator,
        "classification": ClassificationReportGenerator,
        "data_quality": DataQualityReportGenerator,
    }

    @classmethod
    def get_generator(cls, report_type: str) -> BaseReportGenerator:
        generator_cls = cls._generators.get(report_type)
        if not generator_cls:
            raise ValueError(
                f"Unknown report type: {report_type!r}. "
                f"Valid types: {', '.join(cls.get_available_reports())}"
            )
        return generator_cls()

    @classmethod
    def get_available_reports(cls) -> list[str]:
        return list(cls._generators.keys())