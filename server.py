from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow all origins (adjust as needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (adjust as needed)
    allow_headers=["*"],  # Allow all headers (adjust as needed)
)

# Load the pickled model
model_path = 'knn_model.pkl'
model_loaded = pickle.load(open("./knn_model.pkl", "rb"))

@app.post("/api/v1/user/predict")
def predict(data: dict):
    print("Prediction is going on")

    try:
        values = data["values"]
        df = pd.DataFrame([values], columns=data["columns"])

        print(df)
        prediction = model_loaded.predict(df)

        print(prediction)
        prediction_result = prediction.tolist()

        return {"Result": prediction_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=4500)
