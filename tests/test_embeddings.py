"""Embeddings: the client (prefixes, batching, errors) and how RecordDB keeps the vectors in step with the records.
No model is needed: a fake embedder gives deterministic vectors."""
import hashlib
import json
from datetime import datetime
from types import SimpleNamespace

import pytest
from openpyxl import Workbook

from docintel.database import COLUMNS, RecordDB
from docintel.database import embeddings as emb
from docintel.shared import Embedder, EmbeddingError
from docintel.shared.settings import s


class FakeEmbedder:
    """Same surface as Embedder. `fail_after` = how many batches succeed before the model "goes down"."""

    def __init__(self, model="fake-embed", batch_size=2, fail_after=None):
        self.model, self.batch_size, self.fail_after, self.calls, self.texts = model, batch_size, fail_after, 0, []

    def embed_documents(self, texts):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise EmbeddingError("model is down")
        self.calls += 1
        self.texts += list(texts)
        return [self.vector(t) for t in texts]

    def embed_query(self, text):
        return self.vector(text)

    @staticmethod
    def vector(text):
        d = hashlib.sha256(text.encode()).digest()
        return [b / 255 for b in d[:8]]


def rec(i, **kw):
    r = {"record_id": f"R{i:03d}", "chief_complaint": f"complaint {i}", "history_of_present_illness": None,
         "final_diagnosis": f"diagnosis {i}"}
    r.update(kw)
    return r


@pytest.fixture
def db():
    d = RecordDB(":memory:", embedder=FakeEmbedder())
    for i in range(1, 6):
        d.create_record(rec(i))
    yield d
    d.close()


def stored(db):
    return {r[0]: (r[1], r[2]) for r in db.con.execute("SELECT record_id, text_hash, model FROM record_embeddings").fetchall()}


# ---------------------------------------------------------------- the text
def test_embedding_text_uses_the_chosen_fields_in_order_and_skips_empty_ones():
    text = emb.embedding_text({"chief_complaint": " chest pain ", "final_diagnosis": "NSTEMI", "history_of_present_illness": ""})
    assert text == "Chief complaint: chest pain\nFinal diagnosis: NSTEMI"
    assert emb.embedding_text({"chief_complaint": None}) == ""
    assert emb.embedding_text({"plan": "x", "final_diagnosis": "y"}, fields=["plan"]) == "Plan: x"


# ---------------------------------------------------------------- creation and update
def test_embed_pending_creates_one_vector_per_record_and_a_second_run_does_nothing(db):
    report = db.embed_pending()
    assert (report.embedded, report.unchanged) == (5, 0)
    assert set(stored(db)) == {f"R{i:03d}" for i in range(1, 6)}
    calls = db.embedder.calls
    report = db.embed_pending()
    assert (report.embedded, report.unchanged) == (0, 5)
    assert db.embedder.calls == calls  # no model call at all
    assert db.embedding_status() == {"model": "fake-embed", "records": 5, "embedded": 5, "pending": 0}


def test_batches_are_saved_as_they_go_and_progress_is_reported(db):
    seen = []
    db.embed_pending(progress=lambda done, total: seen.append((done, total)))
    assert seen == [(2, 5), (4, 5), (5, 5)]  # batch_size 2


def test_editing_the_text_re_embeds_only_that_record(db):
    db.embed_pending()
    before = stored(db)
    db.update_record({"record_id": "R002", "final_diagnosis": "something else"})
    db.update_record({"record_id": "R003", "plan": "not part of the embedded text"})
    assert db.embedding_status()["pending"] == 1
    assert db.embed_record("R003") is False  # its embedded text did not change
    assert db.embed_record("R002") is True
    after = stored(db)
    assert after["R002"] != before["R002"]
    assert {k: v for k, v in after.items() if k != "R002"} == {k: v for k, v in before.items() if k != "R002"}


def test_deleting_a_record_deletes_its_vector(db):
    db.embed_pending()
    assert db.delete_record("R001") is True
    assert "R001" not in stored(db)
    assert db.delete_record("R001") is False


def test_vectors_of_records_deleted_by_other_means_are_removed_by_embed_pending(db):
    db.embed_pending()
    db.con.execute("DELETE FROM medical_records WHERE record_id = 'R004'")
    report = db.embed_pending()
    assert report.removed == 1 and "R004" not in stored(db)


def test_a_record_without_text_gets_no_vector_and_loses_the_old_one(db):
    db.create_record({"record_id": "EMPTY"})
    db.embed_pending()
    assert "EMPTY" not in stored(db)
    db.update_record({"record_id": "R001", "chief_complaint": None, "final_diagnosis": None}, keep_existing=False)
    assert db.embed_record("R001") is False and "R001" not in stored(db)
    assert db.embed_pending().empty == 2


def test_changing_the_model_re_embeds_everything(db):
    db.embed_pending()
    db._embedder = FakeEmbedder(model="other-model")
    assert db.embedding_status()["pending"] == 5
    assert db.embed_pending().embedded == 5
    assert {m for _, m in stored(db).values()} == {"other-model"}


def test_embed_record_unknown_id(db):
    from docintel import RecordNotFoundError
    with pytest.raises(RecordNotFoundError):
        db.embed_record("NOPE")


def test_when_the_model_goes_down_the_work_done_is_kept_and_a_second_call_finishes(db):
    db._embedder = FakeEmbedder(fail_after=1)  # batch 1 (2 records) works, batch 2 fails
    with pytest.raises(EmbeddingError, match="2 of 5 embedded"):
        db.embed_pending()
    assert len(stored(db)) == 2
    db._embedder = FakeEmbedder()
    assert db.embed_pending().embedded == 3
    assert len(stored(db)) == 5


def test_a_failing_embed_record_leaves_the_record_untouched(db):
    db._embedder = FakeEmbedder(fail_after=0)
    with pytest.raises(EmbeddingError):
        db.embed_record("R001")
    assert db.read_record("R001")["chief_complaint"] == "complaint 1"


def test_vectors_survive_closing_and_reopening(tmp_path):
    path = tmp_path / "d.duckdb"
    with RecordDB(path, embedder=FakeEmbedder()) as d:
        d.create_record(rec(1))
        d.embed_pending()
    with RecordDB(path, embedder=FakeEmbedder()) as d:
        assert d.embedding_status()["pending"] == 0 and len(stored(d)) == 1


# ---------------------------------------------------------------- the Excel pipeline
def xlsx(path, records):
    wb = Workbook()
    ws = wb.active
    ws.append(list(COLUMNS))
    for r in records:
        row = {c: r.get(c) for c in COLUMNS}
        ws.append([datetime.fromisoformat(v) if c in ("dob", "encounter_date") and v else
                   json.dumps(v) if c in ("vital_signs", "lab_results") and v is not None else v for c, v in row.items()])
    wb.save(path)


def test_import_excel_with_embed_creates_the_vectors(tmp_path):
    xlsx(tmp_path / "m.xlsx", [rec(1), rec(2)])
    with RecordDB(":memory:", embedder=FakeEmbedder()) as d:
        assert d.import_excel(tmp_path / "m.xlsx", embed=True) == 2
        assert d.embedding_status() == {"model": "fake-embed", "records": 2, "embedded": 2, "pending": 0}


def test_import_excel_without_embed_leaves_them_pending(tmp_path):
    xlsx(tmp_path / "m.xlsx", [rec(1)])
    with RecordDB(":memory:", embedder=FakeEmbedder()) as d:
        d.import_excel(tmp_path / "m.xlsx")
        assert d.embedding_status()["pending"] == 1 and d.embedder.calls == 0


def test_import_excel_keeps_the_records_when_the_embedding_model_is_down(tmp_path):
    xlsx(tmp_path / "m.xlsx", [rec(1), rec(2)])
    with RecordDB(":memory:", embedder=FakeEmbedder(fail_after=0)) as d:
        with pytest.raises(EmbeddingError):
            d.import_excel(tmp_path / "m.xlsx", embed=True)
        assert d.read_record("R001") is not None and d.embedding_status()["pending"] == 2
        d._embedder = FakeEmbedder()
        assert d.embed_pending().embedded == 2


# ---------------------------------------------------------------- the client
class FakeLiteLLM:
    def __init__(self):
        self.calls = []

    def embedding(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(data=[{"embedding": [1.0, 2.0, 3.0]} for _ in kw["input"]])


@pytest.fixture
def lite(monkeypatch):
    import litellm
    fake = FakeLiteLLM()
    monkeypatch.setattr(litellm, "embedding", fake.embedding)
    return fake


def test_the_default_model_is_the_one_clinic_assistant_used():
    assert s.EMBEDDING_MODEL == "ollama/nomic-embed-text"


def test_nomic_gets_document_and_query_prefixes(lite):
    e = Embedder(batch_size=2)
    e.embed_documents(["a", "b", "c"])
    e.embed_query("q")
    assert [c["input"] for c in lite.calls] == [["search_document: a", "search_document: b"], ["search_document: c"], ["search_query: q"]]


def test_other_models_get_no_prefix(lite):
    Embedder(model="openai/text-embedding-3-small").embed_documents(["a"])
    assert lite.calls[0]["input"] == ["a"]


def test_client_errors_are_wrapped(monkeypatch):
    import litellm

    def boom(**kw):
        raise ConnectionError("refused")
    monkeypatch.setattr(litellm, "embedding", boom)
    with pytest.raises(EmbeddingError, match="failed"):
        Embedder().embed_query("x")


def test_a_wrong_number_of_vectors_is_an_error(monkeypatch):
    import litellm
    monkeypatch.setattr(litellm, "embedding", lambda **kw: SimpleNamespace(data=[]))
    with pytest.raises(EmbeddingError, match="0 vectors for 1 texts"):
        Embedder().embed_query("x")


@pytest.mark.parametrize("model, hint", [("ollama/nomic-embed-text", "ollama pull nomic-embed-text"), ("azure/my-deployment", "name of the deployment")])
def test_a_failed_call_suggests_what_to_check_for_that_provider(model, hint):
    class NotFoundError(Exception):
        pass
    def fail(**kwargs):
        raise NotFoundError("no such deployment")
    import litellm
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(litellm, "embedding", fail)
        with pytest.raises(EmbeddingError) as info:
            Embedder(model=model).embed_query("x")
    message = str(info.value)
    assert hint in message and ("ollama pull" in message) == model.startswith("ollama")
