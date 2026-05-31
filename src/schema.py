import pandera as pa
from pandera import Check, Column, DataFrameSchema

# Schema Pandera para validação do DataFrame de entrada da API.

VALID_YES_NO = Check.isin(["Yes", "No"])
VALID_GENDER = Check.isin(["Male", "Female"])
VALID_INTERNET = Check.isin(["DSL", "Fiber optic", "No"])
VALID_CONTRACT = Check.isin(["Month-to-month", "One year", "Two year"])
VALID_PAYMENT = Check.isin([
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
])
VALID_MULTILINE = Check.isin(["Yes", "No", "No phone service"])
VALID_SERVICE = Check.isin(["Yes", "No", "No internet service"])

input_schema = DataFrameSchema(
    {
        "gender":            Column(str,   VALID_GENDER),
        "SeniorCitizen":     Column(int,   Check.isin([0, 1])),
        "Partner":           Column(str,   VALID_YES_NO),
        "Dependents":        Column(str,   VALID_YES_NO),
        "tenure":            Column(int,   Check.ge(0)),
        "PhoneService":      Column(str,   VALID_YES_NO),
        "MultipleLines":     Column(str,   VALID_MULTILINE),
        "InternetService":   Column(str,   VALID_INTERNET),
        "OnlineSecurity":    Column(str,   VALID_SERVICE),
        "OnlineBackup":      Column(str,   VALID_SERVICE),
        "DeviceProtection":  Column(str,   VALID_SERVICE),
        "TechSupport":       Column(str,   VALID_SERVICE),
        "StreamingTV":       Column(str,   VALID_SERVICE),
        "StreamingMovies":   Column(str,   VALID_SERVICE),
        "Contract":          Column(str,   VALID_CONTRACT),
        "PaperlessBilling":  Column(str,   VALID_YES_NO),
        "PaymentMethod":     Column(str,   VALID_PAYMENT),
        "MonthlyCharges":    Column(float, Check.ge(0)),
        "TotalCharges":      Column(float, Check.ge(0)),
    },
    name="ClienteInputSchema",
    title="Schema de validação do DataFrame de entrada da API de Churn",
)
