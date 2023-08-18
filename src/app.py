from typing import Dict, List
import json
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import models.transformer_model as transformer_model
import models.plagiarism_model as plagiarism_model
from utils.utils import dict_to_list
from fastapi.middleware.cors import CORSMiddleware
from configs.config import configs
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles


BERT = transformer_model.BERTModel()
PM = plagiarism_model.PlagiarismModel(BERT)

app = FastAPI(
    title="Quizzix Plagiarism_API",
    description="This is an API for plagiarism detection in essays. It is based on the sentence-transformers library and the paraphrase-multilingual-MiniLM-L12-v2 model.",
    version="0.1.0", docs_url=None, redoc_url=None
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="../static/swagger-ui-bundle.js",
        swagger_css_url="../static/swagger-ui.css",
    )

@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()



class StudentsDict(BaseModel):
    essays_dict: Dict[str, str]
    cased: bool = False
    
class PlagiarismResponse(BaseModel):
    plagiarism_results: List[Dict[str, Dict[str, float]]]


@app.get(path="/")
def read_root():
    return json.dumps({"message": "Hi"})


@app.post("/plagiarism/predict", response_model=PlagiarismResponse)
def predict_plagiarism(essays_dict: StudentsDict) -> PlagiarismResponse:
    ids, answers = dict_to_list(essays_dict.essays_dict, essays_dict.cased)
    res = PM.predict(answers, ids)
    return {"plagiarism_results": res}


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == '__main__':
    uvicorn.run(app, host=configs['host'], port=configs['port'])