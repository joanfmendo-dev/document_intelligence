"""The copilot page and the hooks that keep embeddings and the catalog up to date when records change."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from docintel import RecordDB
from docintel.shared.embeddings import EmbeddingError
from docintel.web import create_app
from docintel.web.pipeline import Extraction
from test_web import FIELDS, PDF, form_data  # noqa: F401


class Embedder:
    model = "kw"
    batch_size = 10

    def __init__(self):
        self.down, self.calls = False, 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.down:
            raise EmbeddingError("down")
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class LLM:
    def __init__(self):
        self.replies, self.calls, self.messages = [], 0, []

    def run(self, messages, **params):
        self.calls += 1
        self.messages.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return json.dumps(reply)


def rec(i, dept="Cardiology", **more):
    return {**FIELDS, "record_id": f"R{i:03d}", "patient_name": f"Patient {i}", "department": dept, **more}


@pytest.fixture
def env(tmp_path):
    e = SimpleNamespace(db=tmp_path / "db.duckdb", catalog=tmp_path / "cat.yaml", emb=Embedder(), llm=LLM())
    e.app = create_app(db_path=e.db, uploads_dir=tmp_path / "up", allowed_hosts=["testserver"], catalog_path=e.catalog,
                       copilot_llm=e.llm, embedder=e.emb, pipeline=lambda p, on: Extraction(FIELDS, [], 1))
    with TestClient(e.app) as client:
        e.client = client
        yield e


def seed(env, n, dept="Cardiology", embed=False):
    with RecordDB(env.db, embedder=env.emb) as db:
        for i in range(n):
            db.create_record(rec(i, dept))
        if embed:
            db.embed_pending()


def test_the_page_and_the_navigation_link(env):
    page = env.client.get("/copilot").text
    assert "Ask the copilot" in page and 'href="/copilot"' in page
    assert 'href="/copilot"' in env.client.get("/records").text


def test_a_question_is_answered_with_links_to_the_records(env):
    seed(env, 3)
    env.llm.replies = [{"intent": "search", "filters": [{"field": "department", "op": "in", "value": ["Cardiology"]}]}]
    page = env.client.post("/copilot", data={"question": "cardiology patients"}).text
    assert "Understood as" in page and "Patient 1" in page and 'href="/records/R001/edit"' in page
    assert "3 records" in page or "Found 3" in page


def test_the_question_is_escaped_in_the_page(env):
    env.llm.replies = [{"intent": "out_of_scope", "reason": "unrelated"}]
    page = env.client.post("/copilot", data={"question": "<script>alert(1)</script>"}).text
    assert "<script>alert(1)</script>" not in page


def test_a_follow_up_sends_the_previous_plan(env):
    seed(env, 2)
    first = {"intent": "search", "filters": [{"field": "department", "op": "in", "value": ["Cardiology"]}]}
    env.llm.replies = [first, {**first, "filters": first["filters"] + [{"field": "age", "op": "gt", "value": 60}]}]
    page = env.client.post("/copilot", data={"question": "cardiology"}).text
    previous = page.split('name="previous" value="')[1].split('"')[0].replace("&#34;", '"')
    assert json.loads(previous)["filters"][0]["field"] == "department"
    env.client.post("/copilot", data={"question": "only over 60", "previous": previous})
    assert env.llm.calls == 2


def test_next_does_not_call_the_model(env):
    seed(env, 30)
    env.llm.replies = [{"intent": "search", "filters": [{"field": "department", "op": "in", "value": ["Cardiology"]}]}]
    page = env.client.post("/copilot", data={"question": "everyone"}).text
    assert "Page 1 of 2" in page and "Show more" not in page
    plan = page.split('name="plan" value="')[1].split('"')[0].replace("&#34;", '"')
    second = env.client.post("/copilot", data={"question": "everyone", "plan": plan, "page": "2"}).text
    assert env.llm.calls == 1 and "Showing 26–30" in second and "Page 2 of 2" in second


def test_a_model_that_is_down_gives_a_message_not_an_error_page(env):
    from docintel.shared.llm import LLMError
    env.llm.replies = [LLMError("down")]
    r = env.client.post("/copilot", data={"question": "anyone"})
    assert r.status_code == 200 and "not available" in r.text


def test_bad_json_in_the_form_is_a_400(env):
    assert env.client.post("/copilot", data={"question": "x", "previous": "{nope"}).status_code == 400
    assert env.client.post("/copilot", data={"question": "x", "plan": "[1]"}).status_code == 400


def vectors(env):
    with RecordDB(env.db, embedder=env.emb) as db:
        return {r[0] for r in db.con.execute("SELECT record_id FROM record_embeddings").fetchall()}


def test_saving_a_record_creates_its_vector_and_updates_the_catalog(env):
    r = env.client.post("/upload", files={"file": PDF}, follow_redirects=False)
    job = r.headers["location"]
    for _ in range(50):
        if "Review and save" in env.client.get(job).text and 'name="record_id"' in env.client.get(job).text:
            break
    env.client.post(f"{job}/save", data=form_data(FIELDS))
    assert vectors(env) == {"NEW-0001"}
    assert env.catalog.is_file() and "Infectious Diseases" in env.catalog.read_text()


def test_a_save_survives_the_embedding_model_being_down(env):
    seed(env, 1)
    env.emb.down = True
    r = env.client.post("/records/R000/edit", data=form_data(rec(0), chief_complaint="Something new"), follow_redirects=False)
    assert r.status_code == 303
    with RecordDB(env.db) as db:
        assert db.read_record("R000")["chief_complaint"] == "Something new"


def test_editing_refreshes_the_vector_and_deleting_removes_it(env):
    seed(env, 2, embed=True)
    assert vectors(env) == {"R000", "R001"}
    calls = env.emb.calls
    env.client.post("/records/R000/edit", data=form_data(rec(0), chief_complaint="Different complaint"))
    assert env.emb.calls == calls + 1
    env.client.post("/records/R001/delete")
    assert vectors(env) == {"R000"}
    assert "Cardiology" in env.catalog.read_text()


def test_the_chat_endpoint_returns_a_fragment(env):
    seed(env, 3)
    bad = {"intent": "aggregate", "filters": [], "aggregate": {"metric": "count", "group_by": "lab_results.test_name"}}
    env.llm.replies = [bad]
    out = env.client.post("/copilot/ask", data={"question": "How many patients were tested for creatinine?"}).json()
    assert out["kind"] == "aggregate" and "3 records match." in out["html"] and "I cannot group by" in out["html"]
    assert "<html" not in out["html"] and out["plan"]["intent"] == "aggregate" and out["turn"] == 0


CARDIO = {"intent": "search", "filters": [{"field": "department", "op": "in", "value": ["Cardiology"]}]}


def ask_chat(env, question, plan=CARDIO):
    env.llm.replies = [plan]
    return env.client.post("/copilot/ask", data={"question": question}).json()


def test_the_conversation_is_still_there_when_the_user_comes_back(env):
    seed(env, 3)
    ask_chat(env, "cardiology patients")
    ask_chat(env, "<b>again</b>")
    env.client.get("/records")                                   # the user visits another page...
    page = env.client.get("/copilot").text                       # ...and returns
    assert page.count('class="msg me"') == 2 and "cardiology patients" in page and "3 records found." in page
    assert "&lt;b&gt;again&lt;/b&gt;" in page and "<b>again</b>" not in page     # what is shown again is as escaped as the first time
    assert "For example:" not in page                            # it is not an empty page


def test_the_model_sees_the_conversation_the_server_kept(env):
    seed(env, 3)
    first = ask_chat(env, "cardiology patients")
    ask_chat(env, "only over 60")
    sent = json.dumps(env.llm.messages[1])
    assert "cardiology patients" in sent and first["reply"] in sent


def test_clear_conversation_forgets_it_on_the_server(env):
    seed(env, 3)
    ask_chat(env, "cardiology patients")
    r = env.client.post("/copilot/clear", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/copilot"
    page = env.client.get("/copilot").text
    assert 'class="msg me"' not in page and "For example:" in page and "Clear conversation" in page
    ask_chat(env, "again")
    assert "cardiology patients" not in json.dumps(env.llm.messages[-1])      # the model no longer sees the old turns


def test_two_browsers_have_two_conversations(env):
    from fastapi.testclient import TestClient
    seed(env, 3)
    ask_chat(env, "my question")
    other = TestClient(env.client.app)
    assert 'class="msg me"' not in other.get("/copilot").text


def test_the_page_buttons_of_a_long_result(env):
    seed(env, 60)
    out = ask_chat(env, "everyone")
    html = out["html"]
    assert "Page 1 of 3" in html and html.count("data-more") == 4
    buttons = [b for b in html.split("<button") [1:]]
    assert ["disabled" in b.split(">")[0] for b in buttons[:4]] == [True, True, False, False]       # First and Previous off on page 1
    for target, label in (("3", "Last"), ("2", "Previous"), ("1", "First"), ("2", "Next")):
        plan = json.dumps(out["plan"])
        data = {"question": "everyone", "plan": plan, "page": target, "turn": str(out["turn"])}
        page = env.client.post("/copilot/ask", data=data).json()
        assert f"Page {target} of 3" in page["html"] and page["turn"] == out["turn"]
    assert env.llm.calls == 1                                    # turning pages never asks the model
    stored = env.client.get("/copilot").text
    assert "Page 2 of 3" in stored and stored.count('class="msg me"') == 1  # the conversation remembers the page it was left on, and has one turn
    last = env.client.post("/copilot/ask", data={"question": "everyone", "plan": json.dumps(out["plan"]), "page": "3", "turn": "0"}).json()["html"]
    assert [("disabled" in b.split(">")[0]) for b in last.split("<button")[1:5]] == [False, False, True, True]


def test_a_short_result_has_no_page_buttons(env):
    seed(env, 5)
    assert "data-more" not in ask_chat(env, "everyone")["html"]
# ---------------------------------------------------------------- hostile text is escaped in the chat
def test_hostile_text_from_a_record_is_escaped_in_the_chat(env):  
    seed(env, 1)
    env.llm.replies = [{"intent": "search", "filters": [{"field": "department", "op": "in", "value": ["Cardiology"]}]}]
    from docintel import RecordDB
    with RecordDB(env.db, embedder=env.emb) as d:
        d.update_record({"record_id": "R000", "patient_name": "<img src=x onerror=alert(1)>"})
    r = env.client.post("/copilot/ask", data={"question": "cardiology"})
    html = r.json()["html"]
    assert "Found" in html or "R000" in html
    assert "<img src=x" not in html and "&lt;img" in html


def test_a_hostile_question_is_escaped_in_the_chat_reply(env):  
    env.llm.replies = [{"intent": "clarify", "question": "<script>alert(1)</script> which one?"}]
    r = env.client.post("/copilot/ask", data={"question": "<b>hi</b>"})
    html = r.json()["html"]                                          # the only part the page inserts as HTML ("reply" goes back to the model as text)
    assert "<script>" not in html and "&lt;script&gt;" in html
