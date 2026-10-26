"""Consultation audio -> structured prescription -> verified, patient-language summary and audio.

The LLM only extracts. Doses, timings and the patient-facing wording are checked and built
by deterministic code, so a model slip (e.g. "1-0-1" read as three times a day) gets flagged
instead of reaching the patient.
"""
import base64
import glob
import io
import json
import re
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from difflib import SequenceMatcher

from sarvam_client import client

LANGUAGES = {
    "Hindi": "hi-IN", "English": "en-IN", "Bengali": "bn-IN", "Marathi": "mr-IN",
    "Tamil": "ta-IN", "Telugu": "te-IN", "Kannada": "kn-IN", "Malayalam": "ml-IN",
    "Gujarati": "gu-IN", "Punjabi": "pa-IN", "Odia": "od-IN",
}
FOOD_OPTIONS = ["after_food", "before_food", "empty_stomach", "with_food", "not_specified"]
FREQ_OPTIONS = ["daily", "weekly", "as_needed", "other"]


# ---------- 1. Speech to text ----------

def transcribe(audio_path: str, keyterms: list[str] | None = None) -> dict:
    """Batch STT with diarization. Handles long, code-mixed consultations."""
    params = dict(model="saaras:v4", mode="codemix", language_code="unknown",
                  with_diarization=True, num_speakers=2)
    if keyterms:
        params["keyterms"] = [k[:64] for k in keyterms[:50]]
    job = client.speech_to_text_job.create_job(**params)
    job.upload_files(file_paths=[audio_path])
    job.start()
    status = job.wait_until_complete(poll_interval=2, timeout=600)
    if status.job_state != "Completed":
        raise RuntimeError(f"Transcription failed: {status.error_message}")
    with tempfile.TemporaryDirectory() as out_dir:
        job.download_outputs(output_dir=out_dir)
        data = json.load(open(glob.glob(f"{out_dir}/*.json")[0]))
    entries = (data.get("diarized_transcript") or {}).get("entries") or []
    turns = [(f"Speaker {int(e['speaker_id']) + 1}", e["transcript"]) for e in entries]
    return {"transcript": data["transcript"], "turns": turns, "language": data.get("language_code")}


# ---------- 2. LLM extraction ----------

EXTRACT_PROMPT = """You are a clinical scribe for Indian OPD consultations. Read the transcript of a
doctor-patient conversation (may mix Hindi/regional languages and English; speaker labels may be wrong)
and extract ONLY what the doctor explicitly said. Never guess or add medical advice. Use null when not said.

Return only JSON with this shape:
{
  "chief_complaints": [str],
  "vitals": [str, e.g. "BP 150/95", "Fasting sugar 180 mg/dL"],
  "diagnosis": [str],
  "medicines": [{
    "name": str, "strength": str|null, "form": "tablet|capsule|syrup|injection|drops|inhaler|sachet|ointment|other",
    "dose_pattern": "morning-afternoon-night quantities, e.g. 1-0-1, 0-0-1, 1-1-1, 0.5-0-0",
    "dose_unit": "tablet|capsule|ml|tsp|drops|puff|sachet|unit|application",
    "frequency": "daily|weekly|as_needed|other",
    "day_of_week": str|null,
    "food": "after_food|before_food|empty_stomach|with_food|not_specified",
    "duration": str|null,
    "instructions": str|null,
    "evidence": "the exact words from the transcript about this medicine, copied verbatim"
  }],
  "diet": {"avoid": [str], "limit": [str], "prefer": [str]},
  "lifestyle": [str],
  "tests": [str],
  "referrals": [{"specialty": str, "reason": str|null}],
  "follow_up": {"after_days": int|null, "bring": [str], "note": str|null},
  "red_flags": [short symptom phrases, e.g. "chest pain", "dizziness"]
}
For weekly medicines, dose_pattern is the quantity per dose, e.g. 1-0-0. Write everything in English
except copy "evidence" exactly as it appears in the transcript."""


def extract(transcript: str) -> dict:
    last_err = None
    for _ in range(2):
        resp = client.chat.completions(
            model="sarvam-105b-conversations",
            messages=[{"role": "system", "content": EXTRACT_PROMPT},
                      {"role": "user", "content": transcript}],
            max_tokens=6000,
            temperature=0.1,
        )
        text = resp.choices[0].message.content or ""
        try:
            return json.loads(text[text.find("{"): text.rfind("}") + 1])
        except json.JSONDecodeError as e:
            last_err = e
    raise RuntimeError(f"Could not parse prescription from model output: {last_err}")


# ---------- 3. Deterministic decoding + verification ----------

def parse_pattern(pattern: str | None) -> tuple[float, float, float]:
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", pattern or "")]
    nums = (nums + [0, 0, 0])[:3]
    return nums[0], nums[1], nums[2]


SLOT_WORDS = {
    "morning": ["सुबह", "सवेरे", "नाश्ते", "morning", "breakfast", "subah"],
    "afternoon": ["दोपहर", "afternoon", "lunch", "dopahar"],
    "night": ["रात", "सोने", "शाम", "night", "bedtime", "dinner", "evening", "raat"],
}
COUNT_WORDS = {
    1: ["एक बार", "once", "ek baar"],
    2: ["दो बार", "twice", "two times", "do baar"],
    3: ["तीन बार", "thrice", "three times", "teen baar"],
}
FOOD_WORDS = {
    "after_food": ["खाने के बाद", "भोजन के बाद", "after food", "after meal", "khane ke baad"],
    "before_food": ["खाने से पहले", "खाने के पहले", "भोजन से पहले", "before food", "before meal", "नाश्ते से पहले"],
    "empty_stomach": ["खाली पेट", "empty stomach", "khali pet"],
}


def _fuzzy_in(needle: str, hay: str) -> float:
    """Best similarity of needle against same-length windows of hay (0..1)."""
    n, h = " ".join(_words(needle)), " ".join(_words(hay))
    if not n:
        return 0.0
    if n in h:
        return 1.0
    hw, k = h.split(), len(n.split())
    best = 0.0
    for i in range(max(1, len(hw) - k + 1)):
        best = max(best, SequenceMatcher(None, n, " ".join(hw[i:i + k])).ratio())
    return best


def _words(s: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", (s or "").lower()).split()


def _has(text: str, words: list[str]) -> bool:
    t = (text or "").lower()
    return any(w.lower() in t for w in words)


def verify(rx: dict, transcript: str) -> list[dict]:
    """Normalise medicines and return review flags. Mutates rx['medicines'] in place."""
    flags = []
    t_low = transcript.lower()
    for med in rx.get("medicines", []):
        name = med.get("name") or "?"
        if "morning" not in med:
            med["morning"], med["afternoon"], med["night"] = parse_pattern(med.get("dose_pattern"))
        m, a, n = (float(med.get(k) or 0) for k in ("morning", "afternoon", "night"))
        med.setdefault("frequency", "daily")
        if med.get("food") not in FOOD_OPTIONS:
            med["food"] = "not_specified"
        ev = med.get("evidence") or ""

        if not ev:
            continue  # added by the doctor during review

        def flag(msg):
            flags.append({"medicine": name, "issue": msg, "evidence": ev})

        first = _words(name)[0] if _words(name) else ""
        if first and first not in t_low:
            flag("Medicine name was not heard in the recording")
        for num in re.findall(r"\d+", (med.get("strength") or "").replace(",", "")):
            if num not in transcript.replace(",", ""):
                flag(f"Strength '{med.get('strength')}' was not heard in the recording")
                break
        if _fuzzy_in(ev, transcript) < 0.8:
            flag("Quoted evidence does not match the recording")

        if med["frequency"] == "daily":
            said_slots = {s for s, ws in SLOT_WORDS.items() if _has(ev, ws)}
            given_slots = {s for s, v in zip(SLOT_WORDS, (m, a, n)) if v > 0}
            if said_slots and said_slots != given_slots:
                flag(f"Timing mismatch: doctor said {', '.join(sorted(said_slots))}, "
                     f"prescription has {', '.join(sorted(given_slots)) or 'none'}")
            for count, ws in COUNT_WORDS.items():
                if _has(ev, ws) and len(given_slots) != count:
                    flag(f"Doctor said {count}x a day, prescription has {len(given_slots)}x")
            if not given_slots:
                flag("No dose timing (morning/afternoon/night) set")

        said_food = next((f for f, ws in FOOD_WORDS.items() if _has(ev, ws)), None)
        if said_food and med["food"] != said_food and not (said_food == "before_food" and med["food"] == "empty_stomach"):
            flag(f"Food mismatch: doctor said {said_food.replace('_', ' ')}, prescription has {med['food'].replace('_', ' ')}")
        if not med.get("duration") and med["frequency"] != "as_needed":
            flag("Duration not specified")
    return flags


# ---------- 4. Patient-facing text (deterministic templates) ----------

def _qty(q: float, unit: str) -> str:
    q_txt = {0.5: "half", 1: "1", 1.5: "one and a half"}.get(q, f"{q:g}")
    unit = unit or "tablet"
    return f"{q_txt} {unit}" + ("s" if q > 1 and unit in ("tablet", "capsule", "drop", "puff", "sachet") else "")


FOOD_TEXT = {
    "after_food": "after food", "before_food": "before food",
    "empty_stomach": "on an empty stomach", "with_food": "with food", "not_specified": "",
}


def medicine_line(med: dict) -> str:
    unit = med.get("dose_unit") or "tablet"
    slots = [(s, q) for s, q in zip(("in the morning", "in the afternoon", "at night"),
                                     (float(med.get(k) or 0) for k in ("morning", "afternoon", "night"))) if q > 0]
    if med.get("frequency") == "weekly":
        q = slots[0][1] if slots else 1
        when = f"take {_qty(q, unit)} once a week" + (f", every {med['day_of_week'].title()}" if med.get("day_of_week") else "")
    elif med.get("frequency") == "as_needed":
        when = "take only when needed"
    else:
        when = "take " + " and ".join(f"{_qty(q, unit)} {s}" for s, q in slots)
    name = med["name"][:1].upper() + med["name"][1:]
    parts = [f"{name} {med.get('strength') or ''}".strip() + ": " + when]
    if FOOD_TEXT.get(med.get("food")):
        parts.append(FOOD_TEXT[med["food"]])
    if med.get("duration"):
        parts.append(f"for {med['duration']}")
    line = ", ".join(parts) + "."
    if med.get("instructions"):
        line += f" {med['instructions'].rstrip('.')}."
    return line


def patient_sections(rx: dict, visit: date | None = None) -> list[tuple[str, list[str]]]:
    visit = visit or date.today()
    secs = [("Your medicines", [medicine_line(m) for m in rx.get("medicines", [])])]
    diet = rx.get("diet") or {}
    food = []
    if diet.get("avoid"):
        food.append("Do not eat: " + ", ".join(diet["avoid"]) + ".")
    if diet.get("limit"):
        food.append("Eat less: " + ", ".join(diet["limit"]) + ".")
    if diet.get("prefer"):
        food.append("Good to eat: " + ", ".join(diet["prefer"]) + ".")
    food += [s.rstrip(".") + "." for s in rx.get("lifestyle") or []]
    secs.append(("Food and daily routine", food))
    secs.append(("Tests to get done", [t.rstrip(".") + "." for t in rx.get("tests") or []]))
    secs.append(("Specialist doctors you should see", [
        f"Consult {'an' if r['specialty'][:1].lower() in 'aeiou' else 'a'} {r['specialty']}"
        + (f" for {r['reason']}" if r.get("reason") else "") + "."
        for r in rx.get("referrals") or []]))
    fu = rx.get("follow_up") or {}
    fu_lines = []
    if fu.get("after_days"):
        d = visit + timedelta(days=int(fu["after_days"]))
        fu_lines.append(f"Come back to this doctor on {d.strftime('%d %B %Y')}"
                        + (", and bring " + ", ".join(fu["bring"]) if fu.get("bring") else "") + ".")
    if fu.get("note"):
        fu_lines.append(fu["note"].rstrip(".") + ".")
    secs.append(("Your next appointment", fu_lines))
    if rx.get("red_flags"):
        secs.append(("Danger signs. Go to a hospital at once if you have", [
            re.sub(r"^(if|you have)\s+", "", f, flags=re.I).rstrip(".").capitalize() + "." for f in rx["red_flags"]]))
    return [(h, lines) for h, lines in secs if lines]


# ---------- 5. Translation + number check + speech ----------

def _translate(text: str, lang: str) -> str:
    if lang == "en-IN":
        return text
    return client.text.translate(
        input=text, source_language_code="en-IN", target_language_code=lang,
        model="sarvam-translate:v1", numerals_format="international",
    ).translated_text


def _numbers(s: str) -> list[str]:
    return sorted(re.findall(r"\d+", s.replace(",", "")))


def translate_sections(sections, lang: str):
    """Translate every line in parallel; flag lines where numbers changed in translation."""
    items = [h for h, _ in sections] + [ln for _, lines in sections for ln in lines]
    with ThreadPoolExecutor(max_workers=8) as pool:
        out = list(pool.map(lambda t: _translate(t, lang), items))
    lookup = dict(zip(items, out))
    translated, warnings = [], []
    for h, lines in sections:
        tl = [lookup[ln] for ln in lines]
        for src, dst in zip(lines, tl):
            extra = set(_numbers(dst)) - set(_numbers(src))
            missing = {n for n in set(_numbers(src)) - set(_numbers(dst)) if len(n) > 1}
            if extra or missing:
                warnings.append({"line": src, "translated": dst})
        translated.append((lookup[h], tl))
    return translated, warnings


def speak(text: str, lang: str, speaker: str = "ritu") -> bytes:
    """TTS in chunks (API limit), stitched into one WAV."""
    chunks, cur = [], ""
    for sent in re.split(r"(?<=[.।?!])\s+", text):
        if len(cur) + len(sent) > 450 and cur:
            chunks.append(cur)
            cur = ""
        cur += sent + " "
    if cur.strip():
        chunks.append(cur)

    def one(chunk):
        r = client.text_to_speech.convert(text=chunk, language_code=lang, speaker=speaker,
                                          model="bulbul:v3", speech_sample_rate=22050, pace=0.9)
        return base64.b64decode("".join(r.audios))

    with ThreadPoolExecutor(max_workers=8) as pool:
        wavs = list(pool.map(one, chunks))
    buf, params, frames = io.BytesIO(), None, []
    for w in wavs:
        with wave.open(io.BytesIO(w)) as r:
            params = r.getparams()
            frames.append(r.readframes(r.getnframes()))
            frames.append(b"\x00\x00" * int(r.getframerate() * 0.4))
    with wave.open(buf, "wb") as out:
        out.setparams(params)
        out.writeframes(b"".join(frames))
    return buf.getvalue()


def sections_to_text(sections) -> str:
    return "\n".join(f"{h}. " + " ".join(lines) for h, lines in sections)
