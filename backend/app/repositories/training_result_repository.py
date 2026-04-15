"""Repository for TrainingResult database operations."""

from sqlalchemy.orm import Session

from app.models.training_result import TrainingResult


class TrainingResultRepository:
    """Repository for TrainingResult database operations."""

    def __init__(self, db: Session):
        self.db = db

    def add(self, result: TrainingResult) -> TrainingResult:
        """Add a training result to the session."""
        self.db.add(result)
        return result

    def get_recent(self, limit: int = 20) -> list[TrainingResult]:
        """Get recent training results ordered by creation time descending."""
        return (
            self.db.query(TrainingResult)
            .order_by(TrainingResult.created_at.desc())
            .limit(limit)
            .all()
        )
