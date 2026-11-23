// DocBase: the little JavaScript the pages use. Everything else works without it.
(() => {
  // Drag a file onto the upload zone, or pick one: it is sent right away.
  const zone = document.querySelector('[data-dropzone]');
  if (zone) {
    const form = zone.closest('form');
    const input = form.querySelector('input[type=file]');
    const send = () => { if (input.files.length) form.submit(); };
    input.addEventListener('change', send);
    ['dragenter', 'dragover'].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); zone.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); zone.classList.remove('over'); }));
    zone.addEventListener('drop', event => {
      if (event.dataTransfer.files.length) { input.files = event.dataTransfer.files; send(); }
    });
  }

  // Lab results: "Add lab result" adds an empty row, "Remove" removes its row.
  const rows = document.querySelector('[data-lab-rows]');
  if (rows) {
    document.querySelector('[data-add-lab]').addEventListener('click', () => {
      rows.append(document.getElementById('lab-row').content.cloneNode(true));
      rows.lastElementChild.querySelector('input').focus();
    });
    rows.addEventListener('click', event => {
      const button = event.target.closest('[data-remove-lab]');
      if (button) button.closest('tr').remove();
    });
  }

  // Records per page: changing it reloads the list at once (the Filter button does the same).
  document.querySelectorAll('[data-autosubmit]').forEach(select => select.addEventListener('change', () => select.form.submit()));

  // Age = whole years from the date of birth to the encounter date; the server does the same sum when saving.
  const dob = document.getElementById('f-dob'), encounter = document.getElementById('f-encounter_date'), age = document.getElementById('f-age');
  if (dob && encounter && age) {
    const parse = text => { const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(text.trim()); return m ? [+m[1], +m[2], +m[3]] : null; };
    const update = () => {
      const a = parse(dob.value), b = parse(encounter.value);
      const known = a && b && (b[0] > a[0] || b[0] === a[0] && (b[1] > a[1] || b[1] === a[1] && b[2] >= a[2]));
      age.readOnly = !!known;  // calculated: not typed; with a missing or inconsistent date the user can type it
      if (known) age.value = b[0] - a[0] - ((b[1] < a[1] || b[1] === a[1] && b[2] < a[2]) ? 1 : 0);
    };
    [dob, encounter].forEach(input => input.addEventListener('input', update));
    update();
  }

  // ICD-10: the chapter and category follow the code while it is typed (the server looks them up again when saving).
  const icdBox = document.querySelector('[data-icd10]');
  if (icdBox) {
    const code = document.getElementById('f-icd10_code'), chapter = document.getElementById('f-icd10_chapter'), category = document.getElementById('f-icd10_category');
    let last = 0;
    code.addEventListener('change', async () => {
      const ticket = ++last;
      let found = {};
      try { found = await (await fetch('/icd10?code=' + encodeURIComponent(code.value))).json(); } catch (error) { return; }
      if (ticket !== last) return;  // a newer lookup is under way
      chapter.value = found.chapter || '';
      category.value = found.category || '';
      if (found.code) code.value = found.code;
    });
  }

  // Esc closes the delete dialog (it is the "Cancelar" link).
  const cancel = document.querySelector('[data-cancel]');
  if (cancel) document.addEventListener('keydown', event => { if (event.key === 'Escape') location.href = cancel.href; });

  // Copilot chat: each question is sent without leaving the page; the server keeps the conversation (and rebuilds it when you come back).
  const chatForm = document.querySelector('form[data-chat]');
  if (chatForm) {
    const log = document.getElementById('chat');
    const input = chatForm.elements.question;
    const bubble = (text, cls, html) => {
      const el = document.createElement(html ? 'section' : 'div');
      el.className = 'msg ' + cls + (html ? ' card' : '');
      if (html) el.innerHTML = html; else el.textContent = text;
      log.append(el);
      return el;
    };
    const send = async (fields) => {
      const body = new FormData();
      Object.entries(fields).forEach(([k, v]) => body.append(k, v));
      const r = await fetch('/copilot/ask', { method: 'POST', body });
      if (r.status === 401) { location.href = '/login?next=/copilot'; }  // the session ended
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    };
    // The page buttons (First, Previous, Next, Last) of an answer: they fetch that page of the same plan, without asking the model.
    const wire = el => el.querySelectorAll('form[data-more]').forEach(f => f.addEventListener('submit', async event => {
      event.preventDefault();
      try { const out = await send(Object.fromEntries(new FormData(f))); el.innerHTML = out.html; wire(el); el.scrollIntoView({ block: 'nearest' }); } catch (e) { /* keep what is shown */ }
    }));
    log.querySelectorAll('.msg.bot').forEach(wire);
    if (log.lastElementChild) log.lastElementChild.scrollIntoView({ block: 'nearest' });
    chatForm.addEventListener('submit', async event => {
      event.preventDefault();
      const question = input.value.trim();
      if (!question) return;
      bubble(question, 'me');
      input.value = '';
      const pending = bubble('', 'bot', '<p class="sub">Thinking…</p>');
      try {
        const out = await send({ question });
        pending.innerHTML = out.html;
        wire(pending);
      } catch (e) {
        pending.innerHTML = '<p class="err" role="alert">The assistant could not be reached. Please try again.</p>';
      }
      pending.scrollIntoView({ block: 'nearest' });
      input.focus();
    });
  }
})();
