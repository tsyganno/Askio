from sqlalchemy import Column, Integer, String, Text, DateTime, Float, func
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class Query(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    tokens = Column(Integer, default=0)
    duration = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Query(id={self.id}, created_at={self.created_at})>"


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    chunks_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, created_at={self.created_at})>"
