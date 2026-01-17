from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import CreditRiskPredictor

# Initialize app and predictor
app = FastAPI(title="Credit Risk Prediction API")
predictor = CreditRiskPredictor()


# Input schema
class LoanApplication(BaseModel):
    age: int
    income: float
    loanamount: float
    creditscore: int
    monthsemployed: int
    numcreditlines: int
    interestrate: float
    loanterm: int
    dtiratio: float
    education: str
    employmenttype: str
    maritalstatus: str
    hasmortgage: str
    hasdependents: str
    loanpurpose: str
    hascosigner: str


# Health check
@app.get("/")
def health_check():
    return {"status": "ok"}


# Prediction endpoint
@app.post("/predict")
def predict_default(application: LoanApplication):
    result = predictor.predict(application.dict())
    return result
