const form = document.getElementById("upload-form");
const submitBtn = document.getElementById("submit-btn");
const statusSection = document.getElementById("status-section");
const jobStateEl = document.getElementById("job-state");
const bookStatusList = document.getElementById("book-status-list");
const resultsSection = document.getElementById("results-section");
const resultsEl = document.getElementById("results");

let pollTimer = null;

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitBtn.disabled = true;
  resultsSection.hidden = true;
  resultsEl.innerHTML = "";

  try {
    const fd = new FormData();
    for (const file of document.getElementById("candidates").files) {
      fd.append("candidates", file);
    }
    fd.append("priors", document.getElementById("priors").files[0]);
    fd.append("k", document.getElementById("k").value);

    const createResp = await fetch("/api/jobs", { method: "POST", body: fd });
    if (!createResp.ok) throw new Error(`upload failed: ${await createResp.text()}`);
    const job = await createResp.json();

    const processResp = await fetch(`/api/jobs/${job.job_id}/process`, { method: "POST" });
    if (!processResp.ok) throw new Error(`process failed: ${await processResp.text()}`);

    statusSection.hidden = false;
    poll(job.job_id);
  } catch (err) {
    submitBtn.disabled = false;
    alert(err.message);
  }
});

function poll(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    const resp = await fetch(`/api/jobs/${jobId}/status`);
    const status = await resp.json();
    renderStatus(status);
    if (status.state === "done" || status.state === "error") {
      clearInterval(pollTimer);
      submitBtn.disabled = false;
      await renderResults(status);
    }
  }, 1500);
}

function renderStatus(status) {
  jobStateEl.textContent = `Job state: ${status.state}` + (status.error ? ` (${status.error})` : "");
  bookStatusList.innerHTML = "";
  for (const book of status.books) {
    const li = document.createElement("li");
    li.textContent = `${book.filename}: ${book.status}` + (book.error ? ` — ${book.error}` : "");
    bookStatusList.appendChild(li);
  }
}

async function renderResults(status) {
  resultsSection.hidden = false;
  resultsEl.innerHTML = "";
  for (const book of status.books) {
    const card = document.createElement("div");
    card.className = "book-result";

    const heading = document.createElement("h3");
    heading.textContent = book.filename;
    card.appendChild(heading);

    if (book.status === "error") {
      const err = document.createElement("p");
      err.className = "error";
      err.textContent = book.error || "processing failed";
      card.appendChild(err);
      resultsEl.appendChild(card);
      continue;
    }

    const previewResp = await fetch(`/api/jobs/${status.job_id}/books/${book.book_id}/preview`);
    const preview = await previewResp.json();

    const list = document.createElement("ol");
    for (const sample of preview.samples) {
      const item = document.createElement("li");
      item.innerHTML = `<span class="score">score ${sample.score.toFixed(3)}</span><blockquote>${escapeHtml(sample.text)}</blockquote>`;
      list.appendChild(item);
    }
    card.appendChild(list);

    const downloadLink = document.createElement("a");
    downloadLink.href = `/api/jobs/${status.job_id}/books/${book.book_id}/download`;
    downloadLink.textContent = "Download Samples EPUB";
    downloadLink.className = "download-link";
    card.appendChild(downloadLink);

    resultsEl.appendChild(card);
  }
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
