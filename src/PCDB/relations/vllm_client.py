"""Minimal OpenAI-compatible client for the extraction model.

Only the three calls the extraction driver needs: issue one chat completion,
and pull the per-row verdicts and reason codes out of the reply. Written
against the stdlib so the package needs no HTTP dependency - a 25-hour sweep
should not fail on a client library upgrade.
"""
import json
import re
import time
import urllib.error
import urllib.request


def chat(base_url, model, prompt, max_tokens, temperature, timeout, think=False,
         logprobs=False, retries=3):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": temperature, "max_tokens": max_tokens}
    if think:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 20
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        t0 = time.time()
        try:
            req = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
            ch = data["choices"][0]; us = data.get("usage", {}); msg = ch["message"]
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            lp = (ch.get("logprobs") or {}).get("content") or []
            return {"text": msg.get("content") or "", "reasoning": reasoning,
                    "finish": ch.get("finish_reason", ""), "logprobs": lp,
                    "ptok": us.get("prompt_tokens"), "ctok": us.get("completion_tokens"),
                    "latency": round(time.time() - t0, 2), "error": ""}
        except Exception as e:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1)); continue
            detail = ""
            if isinstance(e, urllib.error.HTTPError):
                try: detail = e.read().decode()[:300]
                except Exception: pass
            return {"text": "", "error": f"{type(e).__name__}: {e} {detail}".strip(),
                    "latency": round(time.time() - t0, 2), "ptok": None, "ctok": None, "finish": ""}


VERDICTS = {"yes", "no", "unclear"}

REASON_CODES = {"disease_not_modeled", "entity_not_in_paper", "different_experiment",
                "in_vitro_only", "background_mention", "imaging_or_reagent",
                "induction_agent", "normalization_error", "comorbidity_not_modeled"}


def _verdict(item):
    """Normalise one answer to (label, reason), or (None, "") to discard it.

    The model occasionally swaps the two fields, putting the reason code in
    `label` and prose in `reason`. That happened once in 27,476 relations
    (PMC7546707: label 'different_experiment', reason 'The pulmonary embolism
    model was performed in mice, not rats'). The intent is unambiguous - a
    reason code is only ever emitted for a rejection - so it is mapped back.

    Anything outside the taxonomy is discarded rather than stored. Storing it
    would let an invented category flow into every downstream percentage with
    nothing raising; discarding it puts the row in the driver's `missing` list,
    which is already reported and re-asked. A silent corruption becomes a
    failure mode the pipeline handles.
    """
    label = str(item.get("label", "")).strip().lower()
    reason = str(item.get("reason", "") or "").strip()
    if label in REASON_CODES:
        return "no", label
    if label not in VERDICTS:
        return None, ""
    return label, reason


def parse_labels(text):
    """Extract {row_id: label} from the model's JSON, tolerant of fences/prose."""
    if not text:
        return {}
    # strip code fences
    t = re.sub(r"```(?:json)?", "", text)
    labels = {}
    # try full JSON first
    for candidate in _json_candidates(t):
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        rows = obj.get("labels") if isinstance(obj, dict) else obj
        if isinstance(rows, list):
            for it in rows:
                if isinstance(it, dict) and "row_id" in it and "label" in it:
                    try:
                        label, _ = _verdict(it)
                        if label is not None:
                            labels[int(it["row_id"])] = label
                    except Exception: pass
            if labels:
                return labels
    # fallback: regex scrape of {"row_id":N,..."label":"x"}
    for m in re.finditer(r'"row_id"\s*:\s*(\d+)[^}]*?"label"\s*:\s*"([a-zA-Z]+)"', t):
        labels[int(m.group(1))] = m.group(2).strip().lower()
    return labels


def parse_reasons(text):
    """Extract {row_id: reason} from the model's JSON (reason emitted for 'no')."""
    if not text:
        return {}
    t = re.sub(r"```(?:json)?", "", text)
    reasons = {}
    for cand in _json_candidates(t):
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        rows = obj.get("labels") if isinstance(obj, dict) else obj
        if isinstance(rows, list):
            for it in rows:
                if isinstance(it, dict) and "row_id" in it:
                    try:
                        label, reason = _verdict(it)
                        if label is not None and reason:
                            reasons[int(it["row_id"])] = reason
                    except Exception: pass
            if reasons:
                return reasons
    for m in re.finditer(r'"row_id"\s*:\s*(\d+)[^}]*?"reason"\s*:\s*"([^"]+)"', t):
        reasons[int(m.group(1))] = m.group(2)
    return reasons


def _json_candidates(t):
    # the largest {...} block, plus the whole string
    yield t
    depth = 0; start = None
    for i, c in enumerate(t):
        if c == "{":
            if depth == 0: start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                yield t[start:i+1]
