"""The copilot page: a chat (with JavaScript) or one question per page (without)."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import FormData
from starlette.exceptions import HTTPException

from ..copilot import Copilot
from ..copilot.planner import MAX_TURNS
from .site import Site, read_form

MAX_QUESTION = 500
KEEPS_THE_PLAN = ("clarify", "error", "out_of_scope", "help", "navigate")  # these keep the earlier plan for the next message
EXAMPLES = ("Patients over 60 in Cardiology this year", "How many records per department?", "Who came in with trouble breathing at night?",
            "Average heart rate for patients with chest pain")
BAD_REQUEST = "That request could not be read."


def json_field(form: FormData, name: str, kinds: tuple):
    try:
        value = json.loads(form.get(name) or "null")
    except ValueError:
        raise HTTPException(400, BAD_REQUEST) from None
    if not isinstance(value, kinds):
        raise HTTPException(400, BAD_REQUEST)
    return value


def copilot_routes(site: Site) -> APIRouter:
    router = APIRouter()

    def context(**extra) -> dict:
        return {**dict(active="copilot", question="", answer=None, previous="", plan_json="", examples=EXAMPLES), **extra}

    def ask(form: FormData):
        """The question, the answer, and the plan to show: from a new question (with the chat so far, or just the previous plan)
        or from a plan and a page ('show more': no call to the model)."""
        question = str(form.get("question") or "")[:MAX_QUESTION]
        previous = json_field(form, "previous", (dict, type(None)))
        plan = json_field(form, "plan", (dict, type(None)))
        history = json_field(form, "history", (list, type(None))) or []
        try:
            page = max(1, int(form.get("page") or 1))
        except ValueError:
            raise HTTPException(400, BAD_REQUEST) from None
        history = [h for h in history if isinstance(h, dict)][-MAX_TURNS:]
        with site.open_db() as db:
            copilot = Copilot.from_db(db, catalog_path=site.catalog_file, llm=site.copilot_llm)
            answer = copilot.answer_plan(plan, page) if plan else copilot.ask(question, previous, history=history)
        return question, previous, answer

    @router.get("/copilot")
    def copilot_page(request: Request):
        return site.render(request, "copilot.html", **context())

    @router.post("/copilot")
    def copilot_ask(request: Request, form: FormData = Depends(read_form)):
        """Without JavaScript: one question per page, refined with the previous plan. (With JavaScript the page is a chat: /copilot/ask.)"""
        question, previous, answer = ask(form)
        plan_json = json.dumps(answer.plan) if answer.plan else ""
        carried = json.dumps(previous) if answer.kind in KEEPS_THE_PLAN and previous else plan_json if answer.kind in ("results", "aggregate") else ""
        return site.render(request, "copilot.html", **context(question=question, answer=answer, plan_json=plan_json, previous=carried))

    @router.post("/copilot/ask")
    def copilot_chat(request: Request, form: FormData = Depends(read_form)):
        """One turn of the chat: the answer as an HTML fragment, with what the page needs to send back next time."""
        question, _, answer = ask(form)
        plan_json = json.dumps(answer.plan) if answer.plan else ""
        html = site.templates.get_template("_answer.html").render(request=request, answer=answer, question=question, plan_json=plan_json)
        return JSONResponse({"html": html, "kind": answer.kind, "plan": answer.plan, "reply": answer.summary})

    return router
