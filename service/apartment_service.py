import pickle
from typing import Type
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from domain.domain import ApartmentRequest, ApartmentResponse


class ApartmentService:
    def __init__(self):
        self.path_model = "artifacts/random_forest.pkl"
        self.path_encoder = "artifacts/neighborhood_encoder.pkl"
        self.model = self.load_artifacts(path_model=self.path_model)
        self.le = self.load_artifacts(path_model=self.path_encoder)

    def load_artifacts(self, path_model):
        with open(path_model, "rb") as f:
            artifact = pickle.load(f)
            print(type(artifact))
            print(artifact.get_params())


        return artifact

    def preprocess_input(self, request: ApartmentRequest) -> pd.DataFrame:
        data_dict = {
            'rooms': request.rooms,
            'size': request.size,
            'bathrooms': request.bathrooms,
            'neighbourhood': request.neighbourhood,
            'year_built': request.year_built
        }
        data_dict = {key: [val] for key, val in data_dict.items()}
        data_df = pd.DataFrame.from_dict(data_dict)

        data_df.neighbourhood = data_df.neighbourhood.str.lower()
        print(f'from le : {self.le.classes_}')
        data_df.neighbourhood = self.le.transform(data_df.neighbourhood)
        data_df.neighbourhood = data_df.neighbourhood.astype('category')

        return data_df

    def predict_price(self, request: ApartmentRequest) -> Type[ApartmentResponse]:
        input_df = self.preprocess_input(request)
        apartment_price = self.model.predict(input_df)[0]
        apartment_price = int(apartment_price)

        response = ApartmentResponse
        response.price = apartment_price
        return response


if __name__ == "__main__":

    test_request = ApartmentRequest(rooms=2, size=54, bathrooms=1, neighbourhood="central", year_built=1990)

    apartment_service = ApartmentService()
    res = apartment_service.predict_price(request=test_request)
    print(res.price)

