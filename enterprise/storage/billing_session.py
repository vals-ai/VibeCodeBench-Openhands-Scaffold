from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import DECIMAL, Column, DateTime, Enum, String
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class BillingSession(Base):  # type: ignore
    """
    Represents a Stripe billing session for credit purchases.
    Tracks the status of payment transactions and associated user information.
    """

    __tablename__ = 'billing_sessions'

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    status: Mapped[str | None] = mapped_column(
        Enum(
            'in_progress',
            'completed',
            'cancelled',
            'error',
            name='billing_session_status_enum',
        ),
        default='in_progress',
    )
    billing_session_type: Mapped[str] = mapped_column(
        Enum(
            'DIRECT_PAYMENT',
            'MONTHLY_SUBSCRIPTION',
            name='billing_session_type_enum',
        ),
        nullable=False,
        default='DIRECT_PAYMENT',
    )
    price: Mapped[Decimal] = mapped_column(DECIMAL(19, 4), nullable=False)
    price_code = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),  # type: ignore[attr-defined]
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),  # type: ignore[attr-defined]
    )
