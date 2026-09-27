const $ = (selector, root = document) => root.querySelector(selector);
const questionsRoot = $("#questions");
const template = $("#question-template");
const apiInput = $("#api-url");
const stateInput = $("#state-text");
const runButton = $("#run-decision");
const modelPicker = $("#base-model-picker");
const confidenceFloor = $("#confidence-floor");
const confidenceValue = $("#confidence-value");
let modelMetadata = null;
let modelSwitching = false;
let reviewFloor = .75;
const visualPanel = $(".context-panel");
const questionsPanel = $(".question-section");
const runRow = $(".run-row");
const jsonPanel = $("#json-editor-panel");
const jsonInput = $("#json-request");
const jsonStatus = $("#json-status");
let questionSequence = 0;

const starterQuestion = {
  id: "urgency",
  type: "ordinal",
  instructions: "How urgent is this issue? Choose the highest level supported by the context.",
  options: [
    { id: "0", description: "Routine request; no service impairment." },
    { id: "1", description: "Service is impaired but remains available." },
    { id: "2", description: "Production service is unavailable." },
  ],
};

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function makeOption(option = {}, index = 0, type = "choice") {
  const row = document.createElement("div");
  row.className = "option-row";
  const fixed = type === "boolean";
  row.innerHTML = `
    <span class="option-index">${String(index + 1).padStart(2, "0")}</span>
    <input class="option-id" aria-label="Option ID" maxlength="64" value="${escapeHtml(option.id ?? `option_${index + 1}`)}" ${fixed ? "readonly" : ""}>
    <input class="option-description" aria-label="Option description" maxlength="2048" value="${escapeHtml(option.description ?? "")}" placeholder="Describe this option…">
    <button class="icon-button remove-option" type="button" title="Remove option" aria-label="Remove option" ${fixed ? "disabled" : ""}>×</button>`;
  row.querySelectorAll("input").forEach((input) => input.addEventListener("input", updatePreview));
  row.querySelector(".remove-option").addEventListener("click", () => {
    const list = row.parentElement;
    if (list.children.length <= 2) return;
    row.remove();
    renumberOptions(list);
    updatePreview();
  });
  return row;
}

function renumberOptions(list) {
  [...list.children].forEach((row, index) => {
    row.querySelector(".option-index").textContent = String(index + 1).padStart(2, "0");
  });
  const question = list.closest(".question-card");
  const add = question.querySelector(".add-option");
  add.disabled = list.children.length >= 8 || question.querySelector(".question-type").value === "boolean";
}

function setOptions(card, options, type) {
  const list = card.querySelector(".option-list");
  list.replaceChildren(...options.map((option, index) => makeOption(option, index, type)));
  const fixed = type === "boolean";
  card.querySelector(".add-option").hidden = fixed;
  renumberOptions(list);
}

function addQuestion(data = {}) {
  if (questionsRoot.children.length >= 20) return;
  questionSequence += 1;
  const card = template.content.firstElementChild.cloneNode(true);
  const type = data.type ?? "choice";
  card.querySelector(".question-id").value = data.id ?? `question_${questionSequence}`;
  card.querySelector(".question-type").value = type;
  card.querySelector(".instructions").value = data.instructions ?? "Choose the option best supported by the context.";
  card.querySelector(".question-type-mark").textContent = type.toUpperCase();
  const defaults = data.options ?? (type === "boolean"
    ? [{ id: "false", description: "No" }, { id: "true", description: "Yes" }]
    : type === "ordinal"
      ? [{ id: "0", description: "Low" }, { id: "1", description: "Medium" }, { id: "2", description: "High" }]
      : [{ id: "option_a", description: "First possible answer." }, { id: "option_b", description: "Second possible answer." }]);
  setOptions(card, defaults, type);
  card.querySelector(".question-type").addEventListener("change", (event) => {
    const nextType = event.target.value;
    card.querySelector(".question-type-mark").textContent = nextType.toUpperCase();
    setOptions(card, nextType === "boolean"
      ? [{ id: "false", description: "No" }, { id: "true", description: "Yes" }]
      : nextType === "ordinal"
        ? [{ id: "0", description: "Low" }, { id: "1", description: "Medium" }, { id: "2", description: "High" }]
        : [{ id: "option_a", description: "First possible answer." }, { id: "option_b", description: "Second possible answer." }], nextType);
    updatePreview();
  });
  card.querySelector(".add-option").addEventListener("click", () => {
    const list = card.querySelector(".option-list");
    if (list.children.length >= 8) return;
    const nextIndex = list.children.length;
    const newId = card.querySelector(".question-type").value === "ordinal" ? String(nextIndex) : `option_${nextIndex + 1}`;
    list.append(makeOption({ id: newId, description: "" }, nextIndex, card.querySelector(".question-type").value));
    renumberOptions(list);
    updatePreview();
  });
  card.querySelector(".remove-question").addEventListener("click", () => {
    if (questionsRoot.children.length <= 1) return;
    card.remove();
    updateQuestionNumbers();
    updatePreview();
  });
  card.querySelectorAll("input").forEach((input) => input.addEventListener("input", updatePreview));
  questionsRoot.append(card);
  updateQuestionNumbers();
  updatePreview();
}

function validateRequestShape(request) {
  if (!request || typeof request !== "object" || Array.isArray(request)) throw new Error("The request must be a JSON object.");
  if (Object.keys(request).some((key) => !["state_text", "questions"].includes(key))) throw new Error("Only state_text and questions are allowed at the top level.");
  if (typeof request.state_text !== "string" || request.state_text.length < 1 || request.state_text.length > 65536) throw new Error("state_text must contain 1–65,536 characters.");
  if (!Array.isArray(request.questions) || request.questions.length < 1 || request.questions.length > 20) throw new Error("questions must contain 1–20 questions.");
  const ids = new Set();
  request.questions.forEach((question, index) => {
    const prefix = `Question ${index + 1}`;
    if (!question || typeof question !== "object" || Array.isArray(question)) throw new Error(`${prefix} must be an object.`);
    if (Object.keys(question).some((key) => !["id", "type", "instructions", "options"].includes(key))) throw new Error(`${prefix} has an unsupported field.`);
    if (typeof question.id !== "string" || !question.id.length || question.id.length > 128) throw new Error(`${prefix} needs an ID (up to 128 characters).`);
    if (ids.has(question.id)) throw new Error(`Question ID “${question.id}” is repeated.`);
    ids.add(question.id);
    if (!["choice", "boolean", "ordinal"].includes(question.type)) throw new Error(`${prefix} type must be choice, boolean, or ordinal.`);
    if (typeof question.instructions !== "string" || !question.instructions.length || question.instructions.length > 4096) throw new Error(`${prefix} instructions must contain 1–4,096 characters.`);
    if (!Array.isArray(question.options) || question.options.length < 2 || question.options.length > 8) throw new Error(`${prefix} needs 2–8 options.`);
    const optionIds = new Set();
    question.options.forEach((option, optionIndex) => {
      if (!option || typeof option !== "object" || Array.isArray(option)) throw new Error(`${prefix}, option ${optionIndex + 1} must be an object.`);
      if (Object.keys(option).some((key) => !["id", "description"].includes(key))) throw new Error(`${prefix}, option ${optionIndex + 1} has an unsupported field.`);
      if (typeof option.id !== "string" || !option.id.length || option.id.length > 64 || option.id === "ABSTAIN") throw new Error(`${prefix}, option ${optionIndex + 1} needs a valid ID.`);
      if (optionIds.has(option.id)) throw new Error(`${prefix} repeats option ID “${option.id}”.`);
      optionIds.add(option.id);
      if (typeof option.description !== "string" || !option.description.length || option.description.length > 2048) throw new Error(`${prefix}, option ${optionIndex + 1} needs a description.`);
    });
    if (question.type === "boolean" && (optionIds.size !== 2 || !optionIds.has("false") || !optionIds.has("true"))) throw new Error(`${prefix} boolean option IDs must be “false” and “true”.`);
    if (question.type === "ordinal") {
      const values = [...optionIds].map((id) => /^-?(0|[1-9]\d*)$/.test(id) && id !== "-0" ? Number(id) : NaN);
      if (values.some((value) => !Number.isInteger(value) || value < -2147483648 || value > 2147483647)) throw new Error(`${prefix} ordinal IDs must be 32-bit integers.`);
      const sorted = [...values].sort((a, b) => a - b);
      if (sorted.some((value, i) => i > 0 && value !== sorted[i - 1] + 1)) throw new Error(`${prefix} ordinal IDs must be contiguous.`);
    }
  });
  return request;
}

function setJsonStatus(message, isError = false) {
  jsonStatus.textContent = message;
  jsonStatus.classList.toggle("is-error", isError);
}

function applyJsonRequest() {
  let request;
  try {
    request = validateRequestShape(JSON.parse(jsonInput.value));
  } catch (error) {
    setJsonStatus(error instanceof SyntaxError ? `Invalid JSON: ${error.message}` : error.message, true);
    jsonInput.focus();
    return;
  }
  stateInput.value = request.state_text;
  questionsRoot.replaceChildren();
  questionSequence = 0;
  request.questions.forEach((question) => addQuestion(question));
  updatePreview();
  setJsonStatus(`${request.questions.length} question${request.questions.length === 1 ? "" : "s"} loaded into the builder.`);
  showBuilder();
  questionsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showJsonEditor() {
  const current = readRequest();
  jsonInput.value = JSON.stringify(current, null, 2);
  visualPanel.hidden = true;
  questionsPanel.hidden = true;
  runRow.hidden = true;
  jsonPanel.hidden = false;
  $("#json-mode").classList.add("is-active");
  $("#json-mode").setAttribute("aria-selected", "true");
  $("#builder-mode").classList.remove("is-active");
  $("#builder-mode").setAttribute("aria-selected", "false");
  setJsonStatus("Edit the JSON, then show it in the builder.");
}

function showBuilder() {
  jsonPanel.hidden = true;
  visualPanel.hidden = false;
  questionsPanel.hidden = false;
  runRow.hidden = false;
  $("#builder-mode").classList.add("is-active");
  $("#builder-mode").setAttribute("aria-selected", "true");
  $("#json-mode").classList.remove("is-active");
  $("#json-mode").setAttribute("aria-selected", "false");
}

function updateQuestionNumbers() {
  [...questionsRoot.children].forEach((card, index) => {
    card.querySelector(".question-count").textContent = String(index + 1).padStart(2, "0");
  });
  $("#add-question").disabled = questionsRoot.children.length >= 20;
}

function readRequest() {
  const questions = [...questionsRoot.querySelectorAll(".question-card")].map((card) => ({
    id: card.querySelector(".question-id").value.trim(),
    type: card.querySelector(".question-type").value,
    instructions: card.querySelector(".instructions").value.trim(),
    options: [...card.querySelectorAll(".option-row")].map((row) => ({
      id: row.querySelector(".option-id").value.trim(),
      description: row.querySelector(".option-description").value.trim(),
    })),
  }));
  return { state_text: stateInput.value.trim(), questions };
}

function updatePreview() {
  $("#context-count").textContent = `${stateInput.value.length.toLocaleString()} characters`;
  $("#request-preview").textContent = JSON.stringify(readRequest(), null, 2);
}

function setConnection(connected, message) {
  const badge = $("#connection");
  badge.classList.toggle("is-online", connected);
  badge.classList.toggle("is-offline", !connected);
  badge.querySelector("span").textContent = message;
}

function endpoint() {
  return apiInput.value.trim().replace(/\/+$/, "");
}

function decisionUrl() {
  return `${endpoint().replace(/\/v1\/decide$/i, "")}/v1/decide`;
}

async function checkConnection() {
  if (!endpoint()) return setConnection(false, "Add API address");
  try {
    const response = await fetch(decisionUrl(), { method: "OPTIONS" });
    setConnection(response.ok, response.ok ? "Model connected" : "Model unavailable");
    if (response.ok) await Promise.all([refreshModels(), refreshPolicy()]);
  } catch {
    setConnection(false, "Start local model");
  }
}

function policyUrl() {
  return `${endpoint().replace(/\/v1\/decide$/i, "")}/v1/policy`;
}

function setReviewFloor(value) {
  reviewFloor = value;
  confidenceFloor.value = String(Math.round(value * 100));
  confidenceValue.textContent = `${Math.round(value * 100)}%`;
}

async function refreshPolicy() {
  try {
    const response = await fetch(policyUrl());
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const policy = await response.json();
    if (typeof policy?.confidence_floor !== "number") throw new Error("invalid policy");
    setReviewFloor(policy.confidence_floor);
    confidenceFloor.disabled = false;
  } catch {
    confidenceFloor.disabled = true;
  }
}

async function updateReviewFloor() {
  const requested = Number(confidenceFloor.value) / 100;
  confidenceFloor.disabled = true;
  try {
    const response = await fetch(policyUrl(), {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confidence_floor: requested }),
    });
    const policy = await response.json();
    if (!response.ok) throw new Error(policy?.error?.message ?? `HTTP ${response.status}`);
    setReviewFloor(policy.confidence_floor);
    $("#model-status").textContent = `Review threshold set to ${Math.round(policy.confidence_floor * 100)}%.`;
  } catch (error) {
    setReviewFloor(reviewFloor);
    $("#model-status").textContent = `Could not update review threshold: ${error.message}`;
  } finally { confidenceFloor.disabled = false; }
}

function friendlyModel(modelId) {
  if (modelId === "modernbert-base" || modelId?.includes("modernbert-base")) return "ModernBERT base";
  if (modelId === "qwen3-0.6b" || modelId?.includes("qwen3-0.6b")) return "Qwen3 0.6B";
  return modelId || "Unknown model";
}

async function refreshModels() {
  if (!endpoint()) {
    modelMetadata = null;
    modelPicker.replaceChildren(new Option("API address required", ""));
    modelPicker.disabled = true;
    $("#active-model-name").textContent = "API unavailable";
    $("#model-status").textContent = "Enter the local model API address above.";
    return;
  }
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 2500);
    let response;
    try {
      response = await fetch(`${endpoint().replace(/\/v1\/decide$/i, "")}/v1/models`, { signal: controller.signal });
    } finally { clearTimeout(timer); }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const metadata = await response.json();
    if (!metadata || typeof metadata.active_model_id !== "string" || !Array.isArray(metadata.models)) {
      throw new Error("The API returned an invalid model registry");
    }
    modelMetadata = metadata;
    const active = modelMetadata.active_model_id;
    modelPicker.replaceChildren();
    (modelMetadata.models ?? []).forEach((model) => {
      const option = document.createElement("option");
      option.value = model.model_id;
      option.textContent = `${model.display_name} · ${model.available ? (model.benchmark_candidate ? "Benchmark candidate" : "Ready") : "Unavailable"}`;
      option.disabled = !model.available;
      modelPicker.append(option);
    });
    modelPicker.value = active;
    modelPicker.disabled = false;
    $("#active-model-name").textContent = friendlyModel(active);
    const installed = modelMetadata.models?.filter((model) => model.available).length ?? 0;
    $("#model-status").textContent = installed <= 1 ? "Only installed trained candidates are selectable." : "Model choice applies to the local API; it is not part of the request body.";
  } catch (error) {
    modelMetadata = null;
    modelPicker.replaceChildren(new Option("Registry unavailable", ""));
    modelPicker.disabled = true;
    $("#active-model-name").textContent = "Model API unavailable";
    const reason = error.name === "AbortError" ? "request timed out" : error.message;
    $("#model-status").textContent = `Could not load model registry (${reason}). Check the API address and restart ./start.sh.`;
  }
}

async function activateModel() {
  if (!modelMetadata) return;
  const previous = modelMetadata.active_model_id;
  const chosen = modelPicker.value;
  if (chosen === previous) return;
  modelSwitching = true;
  setBusy(true);
  modelPicker.disabled = true;
  $("#model-status").textContent = "Loading…";
  try {
    const response = await fetch(`${endpoint().replace(/\/v1\/decide$/i, "")}/v1/models/active`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_id: chosen }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload?.error?.message ?? `HTTP ${response.status}`);
    modelMetadata = payload;
    await refreshModels();
    $("#result-content").hidden = true;
    $("#empty-result").hidden = false;
    $("#model-status").textContent = "Ready";
  } catch (error) {
    modelPicker.value = previous;
    $("#model-status").textContent = `Activation failed: ${error.message}`;
  } finally {
    modelSwitching = false;
    modelPicker.disabled = false;
    setBusy(false);
  }
}

function setBusy(busy) {
  runButton.disabled = busy;
  runButton.classList.toggle("is-loading", busy);
  runButton.querySelector("span:first-child").textContent = busy ? "Thinking" : "Run decision";
  runButton.querySelector(".button-arrow").textContent = busy ? "···" : "↗";
}

function showError(status, payload) {
  $("#empty-result").hidden = true;
  const content = $("#result-content");
  content.hidden = false;
  const error = payload?.error ?? { code: `HTTP_${status}`, message: "The model request could not be completed." };
  $("#result-state").textContent = error.code === "LOW_CONFIDENCE" ? "HUMAN REVIEW" : "REQUEST ISSUE";
  content.innerHTML = `
    <div class="error-mark">${error.code === "LOW_CONFIDENCE" ? "↗" : "!"}</div>
    <p class="result-eyebrow">${escapeHtml(error.code ?? "ERROR")}</p>
    <h2>${error.code === "LOW_CONFIDENCE" ? "This one needs\na closer look." : "Couldn't complete\nthat decision."}</h2>
    <p class="result-description">${escapeHtml(error.message ?? "Try again after checking the request and local API.")}</p>
    ${error.question_ids?.length ? `<div class="review-list">Review ${error.question_ids.map(escapeHtml).join(", ")}</div>` : ""}
    ${error.code === "LOW_CONFIDENCE" && payload?.decision ? `<button id="view-scored-output" class="button button-quiet view-output" type="button">View scored output <span>↗</span></button>` : ""}
    ${error.path ? `<small class="error-path">${escapeHtml(error.path)}</small>` : ""}`;
  const viewOutput = $("#view-scored-output", content);
  if (viewOutput) viewOutput.addEventListener("click", () => renderResult(payload.decision, { review: error }));
}

function renderResult(payload, { review = null } = {}) {
  $("#empty-result").hidden = true;
  const content = $("#result-content");
  content.hidden = false;
  $("#result-state").textContent = review ? "REVIEW REQUIRED" : "DECISION COMPLETE";
  const answers = Object.entries(payload.answers ?? {});
  content.innerHTML = `
    <p class="result-eyebrow">${review ? "SCORED OUTPUT · REVIEW REQUIRED" : "MODEL RESPONSE"}</p>
    <h2 class="result-title">Here's what<br>Compass suggests.</h2>
    ${review ? `<p class="result-description">Shown for review. ${escapeHtml(review.question_ids?.join(", ") ?? "One or more answers")} fell below the ${Math.round((review.confidence_floor ?? reviewFloor) * 100)}% threshold.</p>` : ""}
    <div class="answers-list">${answers.map(([id, answer]) => renderAnswer(id, answer)).join("")}</div>
    <div class="result-meta"><span><i class="meta-dot"></i> S1 Compass · ${escapeHtml(friendlyModel(payload.model))}</span><span>${Number(payload.latency_ms).toLocaleString()} ms</span></div>`;
    const active = modelMetadata?.active_model_id;
    if (active && payload.model?.startsWith(`s1-compass-${active}:`)) $("#active-model-name").textContent = friendlyModel(active);
    else if (active) void refreshModels();
}

function renderAnswer(id, answer) {
  const boolean = answer.type === "boolean";
  const selected = boolean ? (answer.value ? "true" : "false") : (answer.choice ?? answer.level);
  const probabilities = boolean
    ? { false: 1 - answer.probability, true: answer.probability }
    : answer.probabilities;
  const confidence = boolean ? Math.max(answer.probability, 1 - answer.probability) : answer.confidence;
  const confidencePercent = Math.round(confidence * 100);
  const label = answer.type === "ordinal" ? `Level ${selected}` : boolean ? (answer.value ? "Yes" : "No") : selected;
  return `<article class="answer-card">
    <div class="answer-heading"><span>${escapeHtml(id)}</span><span class="confidence ${confidence < reviewFloor ? "low" : ""}">${confidencePercent}% confidence</span></div>
    <div class="answer-choice"><span class="choice-check">✓</span><strong>${escapeHtml(label)}</strong><span class="answer-type">${escapeHtml(answer.type)}</span></div>
    <div class="probability-list">${Object.entries(probabilities).map(([option, value]) => `
      <div class="probability-row"><div class="probability-label"><span>${escapeHtml(option)}</span><b>${(value * 100).toFixed(1)}%</b></div>
        <div class="probability-track"><i style="width:${Math.max(0, Math.min(100, value * 100))}%"></i></div></div>`).join("")}</div>
  </article>`;
}

async function runDecision() {
  if (modelSwitching) return;
  await refreshModels();
  if (modelMetadata?.active_model_id) {
    modelPicker.value = modelMetadata.active_model_id;
    $("#active-model-name").textContent = friendlyModel(modelMetadata.active_model_id);
  }
  const request = readRequest();
  updatePreview();
  if (!request.state_text || request.questions.some((q) => !q.id || !q.instructions || q.options.some((option) => !option.id || !option.description))) {
    showError(422, { error: { code: "INVALID_SCHEMA", message: "Fill in the context, question instructions, option IDs, and descriptions before running." } });
    return;
  }
  setBusy(true);
  $("#result-state").textContent = "SCORING OPTIONS";
  $("#empty-result").hidden = true;
  $("#result-content").hidden = true;
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 3000);
    let response;
    try {
      response = await fetch(decisionUrl(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal: controller.signal,
      });
    } finally { clearTimeout(timer); }
    console.log(endpoint())
    let payload;
    try { payload = await response.json(); }
    catch {
      showError(502, { error: { code: "INVALID_RESPONSE", message: `The API at ${decisionUrl()} did not return JSON. Check that the address ends at /v1/decide and that the S1 Compass API is running.` } });
      return;
    }
    if (response.ok) renderResult(payload);
    else if (payload?.error?.code === "ACTIVE_MODEL_MISMATCH") await refreshModels();
    else showError(response.status, payload);
  } catch (error) {
    const timedOut = error.name === "AbortError";
    showError(timedOut ? 504 : 0, { error: {
      code: timedOut ? "TIMEOUT" : "API_UNREACHABLE",
      message: timedOut ? `The request to ${decisionUrl()} took too long. Check that the local model is running.` : `Couldn't reach ${decisionUrl()} (${error.message || "network or browser origin error"}). Open the client and API on the same machine, then check that both servers are running.`,
    } });
    setConnection(false, "Start local model");
  } finally {
    setBusy(false);
    checkConnection();
  }
}

$("#add-question").addEventListener("click", () => addQuestion());
$("#builder-mode").addEventListener("click", showBuilder);
$("#json-mode").addEventListener("click", showJsonEditor);
$("#apply-json").addEventListener("click", applyJsonRequest);
jsonInput.addEventListener("input", () => setJsonStatus("JSON edited. Apply it to update the builder."));
runButton.addEventListener("click", runDecision);
modelPicker.addEventListener("change", activateModel);
confidenceFloor.addEventListener("change", updateReviewFloor);
stateInput.addEventListener("input", updatePreview);
apiInput.value = localStorage.getItem("s1-compass-api-url") ?? apiInput.value;
apiInput.addEventListener("change", () => {
  localStorage.setItem("s1-compass-api-url", endpoint());
  checkConnection();
});
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") runDecision();
});

addQuestion(starterQuestion);
updatePreview();
checkConnection();
setInterval(checkConnection, 12000);
