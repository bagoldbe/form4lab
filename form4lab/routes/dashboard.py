from fastapi import APIRouter, Request

from form4lab.templating import templates

router = APIRouter()


@router.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")
