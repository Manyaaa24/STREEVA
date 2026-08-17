from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from .database import Base

class Trip(Base):
    __tablename__ = "trips"

    id = Column(String, primary_key=True, index=True)
    label = Column(String, index=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)

    points = relationship("TripPoint", back_populates="trip", cascade="all, delete-orphan")

class TripPoint(Base):
    __tablename__ = "trip_points"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    trip_id = Column(String, ForeignKey("trips.id"))
    timestamp = Column(DateTime, index=True)
    lat = Column(Float)
    lng = Column(Float)
    accel_x = Column(Float)
    accel_y = Column(Float)
    accel_z = Column(Float)

    trip = relationship("Trip", back_populates="points")
