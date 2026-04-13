from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class EducationLevel(Base):
    __tablename__ = 'education_levels'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    customers: Mapped[list['Customer']] = relationship(back_populates='education_level')

    def __repr__(self) -> str:
        return f'<EducationLevel {self.name!r}>'


class MaritalStatus(Base):
    __tablename__ = 'marital_statuses'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    customers: Mapped[list['Customer']] = relationship(back_populates='marital_status')

    def __repr__(self) -> str:
        return f'<MaritalStatus {self.name!r}>'