from pydantic import BaseModel


class CartesianCoordinate2D(BaseModel):
    x: float
    y: float


class BoundingBox(BaseModel):
    box_coordinates_min: CartesianCoordinate2D
    box_coordinates_max: CartesianCoordinate2D
    box_coordinates_center: CartesianCoordinate2D
    label: str
    score: float
