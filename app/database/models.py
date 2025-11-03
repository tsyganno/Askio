from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    func,
    Index,
)


Base = declarative_base()


class QueryHistory(Base):
    """
    История вопросов/ответов (для аналитики, метрик, ретрейсинга).
    Сохраняем вопрос, ответ, количество токенов и модель.
    """
    __tablename__ = "query_history"
    __table_args__ = (
        Index("ix_query_history_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    tokens = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)  # время ответа в ms
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<QueryHistory(id={self.id}, model={self.model}, created_at={self.created_at})>"


class Document(Base):
    """
    Хранит метаинформацию о загруженном документе (pdf/txt/md).
    Текст сам по себе сохраняется в DocumentChunk.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    chunks_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # relationship -> DocumentChunk.document
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan", lazy="selectin")

    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', created_at={self.created_at})>"


class DocumentChunk(Base):
    """
    Один текстовый чанк/фрагмент из документа.
    Полезно для векторизации / поиска.
    """
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_document_chunks_document_id", "document_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # relationship -> Document.chunks
    document = relationship("Document", back_populates="chunks", lazy="joined")

    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, created_at={self.created_at})>"
