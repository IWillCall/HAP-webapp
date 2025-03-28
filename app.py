from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import json
from utils.preprocessing import get_preprocessor
from models.model import get_model
from utils.process_request import validate_request, decode_request
from utils.utils import get_risk_level

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

with open("data/questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

preprocessor = get_preprocessor()
model = get_model()

@app.get("/", response_class=HTMLResponse)
async def get_risk(request: Request):
    return templates.TemplateResponse("risk.html", {"request": request, "questions": questions, "result": None, "errors": {}, "form_data": {}})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.post("/api/risk")
async def api_risk(request: Request):
    form = await request.form()
    valid_res = validate_request(form)
    if len(valid_res['errors']) > 0:
        return JSONResponse({"errors": valid_res["errors"]})
    request_df = decode_request(form)
    request_enc = preprocessor.transform(request_df)
    risk = model.predict_proba(request_enc)[0][1]
    level, color = get_risk_level(risk)
    return JSONResponse({ "level": level, "color": color})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
