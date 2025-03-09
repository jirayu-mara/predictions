from typing import Type

from fastapi import FastAPI
from domain.domain import ApartmentRequest, ApartmentResponse
from service.apartment_service import ApartmentService

price_app = FastAPI()

# Add a simple root route to avoid 404 errors
@price_app.get("/")
async def root():
    return {"message": "Your FastAPI app is running on Render!"}
@price_app.post("/predict")
async def predict_price(request: ApartmentRequest) -> ApartmentResponse:
    return ApartmentService().predict_price(request=request)


