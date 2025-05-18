"""
Database models for the Name Matching application.

This module defines SQLAlchemy models for the database tables.
"""

from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Base class for all models
Base = declarative_base()


class PersonRecord(Base):
    """
    Model for a person record in the database.
    
    This model can be used for both table sets by specifying the table_name.
    """
    
    __tablename__ = "person_records"
    
    # Allow using this model for different tables
    __table_args__ = {"extend_existing": True}
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Original record ID from the source table
    hh_id = Column(String(50), nullable=False, index=True)
    
    # Name fields
    first_name = Column(String(100), nullable=False, index=True)
    middle_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=False, index=True)
    
    # Original middle_name_last_name field
    middle_name_last_name = Column(String(200), nullable=True)
    
    # Additional fields
    birthdate = Column(Date, nullable=True, index=True)
    province_name = Column(String(100), nullable=True, index=True)
    city_name = Column(String(100), nullable=True, index=True)
    barangay_name = Column(String(100), nullable=True, index=True)
    
    # Metadata
    source_table = Column(String(100), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    match_results_as_record1 = relationship(
        "MatchResult", 
        foreign_keys="MatchResult.record1_id",
        back_populates="record1"
    )
    match_results_as_record2 = relationship(
        "MatchResult", 
        foreign_keys="MatchResult.record2_id",
        back_populates="record2"
    )
    
    def __repr__(self) -> str:
        return (
            f"<PersonRecord(id={self.id}, "
            f"hh_id='{self.hh_id}', "
            f"name='{self.first_name} {self.last_name}')>"
        )
    
    @classmethod
    def from_dict(cls, data: dict, source_table: str) -> "PersonRecord":
        """
        Create a PersonRecord from a dictionary.
        
        Args:
            data: Dictionary with record data
            source_table: Name of the source table
            
        Returns:
            PersonRecord instance
        """
        # Handle birthdate conversion
        birthdate_val = data.get("birthdate")
        if birthdate_val and isinstance(birthdate_val, str):
            try:
                birthdate_val = datetime.strptime(birthdate_val, "%Y-%m-%d").date()
            except ValueError:
                birthdate_val = None
        
        # Create record
        return cls(
            hh_id=data.get("hh_id", str(data.get("id", ""))),
            first_name=data.get("first_name", ""),
            middle_name=data.get("middle_name", ""),
            last_name=data.get("last_name", ""),
            middle_name_last_name=data.get("middle_name_last_name", ""),
            birthdate=birthdate_val,
            province_name=data.get("province_name", ""),
            city_name=data.get("city_name", ""),
            barangay_name=data.get("barangay_name", ""),
            source_table=source_table,
        )
    
    def to_dict(self) -> dict:
        """
        Convert the record to a dictionary.
        
        Returns:
            Dictionary representation of the record
        """
        birthdate_str = None
        if self.birthdate:
            birthdate_str = self.birthdate.strftime("%Y-%m-%d")
            
        return {
            "id": self.id,
            "hh_id": self.hh_id,
            "first_name": self.first_name,
            "middle_name": self.middle_name,
            "last_name": self.last_name,
            "middle_name_last_name": self.middle_name_last_name,
            "birthdate": birthdate_str,
            "province_name": self.province_name,
            "city_name": self.city_name,
            "barangay_name": self.barangay_name,
            "source_table": self.source_table,
        }


class MatchResult(Base):
    """Model for storing match results between two records."""
    
    __tablename__ = "match_results"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys to the matched records
    record1_id = Column(Integer, ForeignKey("person_records.id"), nullable=False)
    record2_id = Column(Integer, ForeignKey("person_records.id"), nullable=False)
    
    # Match details
    score = Column(Float, nullable=False)
    classification = Column(String(20), nullable=False)  # match, non_match, manual_review
    
    # Component scores
    score_first_name = Column(Float, nullable=True)
    score_middle_name = Column(Float, nullable=True)
    score_last_name = Column(Float, nullable=True)
    score_full_name_sorted = Column(Float, nullable=True)
    score_birthdate = Column(Float, nullable=True)
    score_geography = Column(Float, nullable=True)
    
    # Additional information
    notes = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    record1 = relationship(
        "PersonRecord", 
        foreign_keys=[record1_id],
        back_populates="match_results_as_record1"
    )
    record2 = relationship(
        "PersonRecord", 
        foreign_keys=[record2_id],
        back_populates="match_results_as_record2"
    )
    
    # Ensure we don't have duplicate matches
    __table_args__ = (
        UniqueConstraint("record1_id", "record2_id", name="uix_match_result"),
    )
    
    def __repr__(self) -> str:
        return (
            f"<MatchResult(id={self.id}, "
            f"record1_id={self.record1_id}, "
            f"record2_id={self.record2_id}, "
            f"score={self.score}, "
            f"classification='{self.classification}')>"
        )
    
    @classmethod
    def from_dict(cls, data: dict) -> "MatchResult":
        """
        Create a MatchResult from a dictionary.
        
        Args:
            data: Dictionary with match result data
            
        Returns:
            MatchResult instance
        """
        return cls(
            record1_id=data["record1_id"],
            record2_id=data["record2_id"],
            score=data["score"],
            classification=data["classification"],
            score_first_name=data.get("score_first_name"),
            score_middle_name=data.get("score_middle_name"),
            score_last_name=data.get("score_last_name"),
            score_full_name_sorted=data.get("score_full_name_sorted"),
            score_birthdate=data.get("score_birthdate"),
            score_geography=data.get("score_geography"),
            notes=data.get("notes"),
        )
    
    def to_dict(self) -> dict:
        """
        Convert the match result to a dictionary.
        
        Returns:
            Dictionary representation of the match result
        """
        return {
            "id": self.id,
            "record1_id": self.record1_id,
            "record2_id": self.record2_id,
            "score": self.score,
            "classification": self.classification,
            "score_first_name": self.score_first_name,
            "score_middle_name": self.score_middle_name,
            "score_last_name": self.score_last_name,
            "score_full_name_sorted": self.score_full_name_sorted,
            "score_birthdate": self.score_birthdate,
            "score_geography": self.score_geography,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
