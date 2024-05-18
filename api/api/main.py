import pickle
import pandas as pd
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="Zoo Animal CLassification",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

model = pickle.load(
    open('model.pkl', 'rb')
)


@app.get("/")
def read_root(text: str = ""):
    if not text:
        return f"Try to append ?text=something in the URL!"
    else:
        return text


class Animal(BaseModel):
    hair: int
    feathers: int
    eggs: int
    milk: int
    airborne: int
    aquatic: int
    predator: int
    toothed: int
    backbone: int
    breathes: int
    venomous: int
    fins: int
    legs: int
    tail: int
    domestic: int
    catsize: int


@app.post("/predict/")
def predict(animals: List[Animal]) -> List[str]:
    X = pd.DataFrame([dict(animal) for animal in animals])
    y_pred = model.predict(X)
    return list(y_pred)
