"""Pydantic request models for the LocalMate API."""

from typing import Optional

from pydantic import BaseModel


class PlanRequest(BaseModel):
    user_input: str
    session_id: Optional[str] = None
    room_id: Optional[str] = None
    client_id: Optional[str] = None
    speaker: Optional[str] = None
    internal_room_replan: bool = False
    latest_requirement_input: Optional[str] = None


class RoomRequest(BaseModel):
    client_id: str
    speaker: Optional[str] = None


class RoomConfirmRequest(BaseModel):
    client_id: str
    speaker: Optional[str] = None
    plan_revision: Optional[int] = None


class ConfirmRequest(BaseModel):
    session_id: str
    confirmed: bool


class CouponReserveRequest(BaseModel):
    session_id: str
    place_name: str


class PlaceReserveRequest(BaseModel):
    session_id: str
    place_name: str
