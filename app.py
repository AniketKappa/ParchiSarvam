"""Parchi: the prescription that listens. Run with: streamlit run app.py"""
import html
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

import rx_pipeline as rp

SAMPLE = Path(__file__).parent / "samples" / "consultation_hi.wav"
MED_COLS = ["name", "strength", "morning", "afternoon", "night", "dose_unit",
            "frequency", "day_of_week", "food", "duration", "instructions", "evidence"]

st.set_page_config(page_title="Parchi", page_icon="🩺", layout="wide")

# ---------- Sidebar: visit details ----------
with st.sidebar:
    st.header("Visit details")
    doctor = st.text_input("Doctor", "Dr. A. Sharma, MD")
    clinic = st.text_input("Clinic", "Sharma Clinic, Lucknow")
    patient = st.text_input("Patient name", "Kamla Devi")
    c1, c2 = st.columns(2)
    age = c1.text_input("Age", "58")
    sex = c2.selectbox("Sex", ["F", "M", "Other"])
    visit = st.date_input("Visit date", date.today())
    lang_name = st.selectbox("Patient's language", list(rp.LANGUAGES), index=0)
    keyterms = st.text_area(
        "Medicines you often prescribe (helps recognition)",
        "Metformin, Amlodipine, Pantoprazole, Telmisartan, Atorvastatin, Glimepiride",
        help="Sent to Saaras as keyterms so brand and generic names are transcribed correctly.",
    )

st.title("🩺 Parchi")
st.caption("Talk to your patient as usual. Parchi listens and writes a complete, verified prescription, "
           "then explains it to the patient in their own language.")

# ---------- Step 1: audio ----------
st.subheader("1 · Consultation audio")
tab_rec, tab_up, tab_sample = st.tabs(["🎙️ Record", "📁 Upload", "▶️ Sample consultation"])
audio_bytes, audio_name = None, None
with tab_rec:
    rec = st.audio_input("Record the consultation")
    if rec:
        audio_bytes, audio_name = rec.getvalue(), "recording.wav"
with tab_up:
    up = st.file_uploader("Audio file", type=["wav", "mp3", "m4a", "ogg", "webm", "aac", "flac"])
    if up:
        audio_bytes, audio_name = up.getvalue(), up.name
with tab_sample:
    st.write("An 83-second Hindi-English consultation: a new diabetes and BP patient.")
    st.audio(str(SAMPLE))
    if st.checkbox("Use this sample"):
        audio_bytes, audio_name = SAMPLE.read_bytes(), SAMPLE.name

if st.button("Write the Parchi", type="primary", disabled=audio_bytes is None):
    with st.status("Working…", expanded=True) as status:
        with tempfile.NamedTemporaryFile(suffix=Path(audio_name).suffix or ".wav", delete=False) as f:
            f.write(audio_bytes)
        st.write("Transcribing with Saaras (speaker-separated, code-mixed)…")
        terms = [t.strip() for t in keyterms.split(",") if t.strip()]
        tr = rp.transcribe(f.name, terms)
        st.write("Extracting the prescription with Sarvam-105B…")
        rx = rp.extract(tr["transcript"])
        st.write("Cross-checking every medicine against the recording…")
        rp.verify(rx, tr["transcript"])
        status.update(label="Draft ready for your review", state="complete")
    st.session_state.update(tr=tr, rx=rx, approved=False)
    for k in ("audio", "meds", "guide_key"):
        st.session_state.pop(k, None)

if "rx" not in st.session_state:
    st.stop()

tr, rx = st.session_state.tr, st.session_state.rx


def lines(items):
    return "\n".join(items or [])


def unlines(text):
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


# ---------- Step 2: doctor review ----------
st.subheader("2 · Review and edit")
with st.expander("Transcript", expanded=False):
    for spk, text in tr["turns"] or [("", tr["transcript"])]:
        st.markdown(f"**{spk}** {text}")

diagnosis = st.text_input("Diagnosis", ", ".join(rx.get("diagnosis") or []))
v = rx.get("vitals") or []
vitals = st.text_input("Vitals", ", ".join(v if isinstance(v, list) else (f"{k}: {x}" for k, x in v.items())))

st.markdown("**Medicines** · quantities per dose for morning / afternoon / night")
df = pd.DataFrame(rx.get("medicines") or [], columns=MED_COLS)
edited = st.data_editor(
    df, num_rows="dynamic", use_container_width=True, hide_index=True, key="meds",
    column_config={
        "name": st.column_config.TextColumn("Medicine", required=True),
        "strength": "Strength",
        "morning": st.column_config.NumberColumn("Morning", min_value=0, step=0.5),
        "afternoon": st.column_config.NumberColumn("Afternoon", min_value=0, step=0.5),
        "night": st.column_config.NumberColumn("Night", min_value=0, step=0.5),
        "dose_unit": st.column_config.SelectboxColumn(
            "Unit", options=["tablet", "capsule", "ml", "tsp", "drops", "puff", "sachet", "unit", "application"]),
        "frequency": st.column_config.SelectboxColumn("Frequency", options=rp.FREQ_OPTIONS),
        "day_of_week": "Day (weekly)",
        "food": st.column_config.SelectboxColumn("Food", options=rp.FOOD_OPTIONS),
        "duration": "Duration",
        "instructions": "Notes",
        "evidence": st.column_config.TextColumn("What the doctor said", disabled=True, width="large"),
    },
)
meds = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in edited.to_dict("records") if row.get("name")]
rx_final = dict(rx, medicines=meds)
flags = rp.verify(rx_final, tr["transcript"])

if flags:
    st.error(f"⚠️ {len(flags)} item(s) don't match what was said in the consultation. Please check:")
    for fl in flags:
        st.markdown(f"- **{fl['medicine']}**: {fl['issue']}  \n  <small>Heard: “{html.escape(fl['evidence'])}”</small>",
                    unsafe_allow_html=True)
else:
    st.success("✓ Every medicine's name, strength, timing and food instruction matches the recording.")

c1, c2, c3 = st.columns(3)
diet = rx.get("diet") or {}
avoid = c1.text_area("Avoid (one per line)", lines(diet.get("avoid")))
limit = c2.text_area("Eat less", lines(diet.get("limit")))
prefer = c3.text_area("Good to eat", lines(diet.get("prefer")))
c1, c2, c3 = st.columns(3)
lifestyle = c1.text_area("Lifestyle advice", lines(rx.get("lifestyle")))
tests = c2.text_area("Tests", lines(rx.get("tests")))
referrals = c3.text_area("Referrals (specialty — reason)", lines(
    f"{r['specialty']} — {r['reason']}" if r.get("reason") else r["specialty"] for r in rx.get("referrals") or []))
c1, c2, c3 = st.columns(3)
fu = rx.get("follow_up") or {}
fu_days = c1.number_input("Follow-up after (days)", 0, 365, int(fu.get("after_days") or 0))
fu_bring = c2.text_input("Bring to follow-up", ", ".join(fu.get("bring") or []))
red_flags = c3.text_area("Danger signs", lines(rx.get("red_flags")))

rx_final.update(
    diagnosis=[d.strip() for d in diagnosis.split(",") if d.strip()],
    diet={"avoid": unlines(avoid), "limit": unlines(limit), "prefer": unlines(prefer)},
    lifestyle=unlines(lifestyle), tests=unlines(tests), red_flags=unlines(red_flags),
    referrals=[dict(zip(("specialty", "reason"), [p.strip() for p in r.split("—", 1)] + [None]))
               for r in unlines(referrals)],
    follow_up={"after_days": fu_days or None, "bring": [b.strip() for b in fu_bring.split(",") if b.strip()]},
)

label = f"I have reviewed this prescription{' and the ' + str(len(flags)) + ' warning(s)' if flags else ''}"
if not st.checkbox(label, key="approved"):
    st.stop()

# ---------- Step 3: outputs ----------
st.subheader("3 · Prescription and patient guide")


def dose_text(m):
    q = [float(m.get(k) or 0) for k in ("morning", "afternoon", "night")]
    base = "-".join(f"{x:g}" for x in q)
    if m.get("frequency") == "weekly":
        return f"{next((x for x in q if x), 1):g} {m.get('dose_unit') or ''} weekly" + (f" ({m['day_of_week'].title()})" if m.get("day_of_week") else "")
    if m.get("frequency") == "as_needed":
        return "SOS"
    return base


def rx_html(rx):
    e = html.escape
    rows = "".join(
        f"<tr><td><b>{e(m['name'].title())}</b> {e(m.get('strength') or '')}</td><td>{e(dose_text(m))}</td>"
        f"<td>{e((m.get('food') or '').replace('_', ' ').replace('not specified', '—'))}</td>"
        f"<td>{e(m.get('duration') or '—')}</td><td>{e(m.get('instructions') or '')}</td></tr>"
        for m in rx["medicines"])

    def block(title, items):
        return f"<h3>{title}</h3><ul>{''.join(f'<li>{e(i)}</li>' for i in items)}</ul>" if items else ""

    d = rx["diet"]
    advice = [f"Avoid: {', '.join(d['avoid'])}"] * bool(d["avoid"]) + [f"Eat less: {', '.join(d['limit'])}"] * bool(d["limit"]) \
        + [f"Prefer: {', '.join(d['prefer'])}"] * bool(d["prefer"]) + rx["lifestyle"]
    refs = [f"{r['specialty']}" + (f" — {r['reason']}" if r.get("reason") else "") for r in rx["referrals"]]
    fu = rx["follow_up"]
    fu_line = []
    if fu.get("after_days"):
        fu_line = [f"Review on {(visit + timedelta(days=fu['after_days'])).strftime('%d %b %Y')}"
                   + (f" with {', '.join(fu['bring'])}" if fu.get("bring") else "")]
    return f"""<div style="font-family:Georgia,serif;max-width:820px;margin:auto;padding:28px;border:1px solid #ccc;
background:#fff;color:#111;border-radius:6px">
<div style="display:flex;justify-content:space-between;border-bottom:3px double #333;padding-bottom:8px">
<div><div style="font-size:22px;font-weight:bold">{e(doctor)}</div><div>{e(clinic)}</div></div>
<div style="text-align:right">{visit.strftime('%d %b %Y')}<br>{e(patient)} · {e(age)} {e(sex)}</div></div>
<p><b>Diagnosis:</b> {e(', '.join(rx['diagnosis']) or '—')}<br><b>Vitals:</b> {e(vitals or '—')}</p>
<div style="font-size:30px;font-style:italic">℞</div>
<table style="width:100%;border-collapse:collapse" border="1" cellpadding="6">
<tr style="background:#f2f2f2"><th>Medicine</th><th>Dose (M-A-N)</th><th>Food</th><th>Duration</th><th>Notes</th></tr>
{rows}</table>
{block('Advice', advice)}{block('Investigations', rx['tests'])}{block('Referrals', refs)}
{block('Follow-up', fu_line)}{block('Return immediately if', rx['red_flags'])}
<p style="margin-top:32px;text-align:right">______________________<br>{e(doctor)}</p>
<p style="font-size:11px;color:#777">Drafted by Parchi from the recorded consultation and reviewed by the doctor.</p>
</div>"""


tab_rx, tab_pt = st.tabs(["📄 Prescription", f"🗣️ Patient guide ({lang_name})"])
with tab_rx:
    page = rx_html(rx_final)
    st.html(page)
    st.download_button("Download printable prescription", f"<html><body>{page}</body></html>",
                       file_name=f"parchi_{patient.replace(' ', '_')}.html", mime="text/html")

with tab_pt:
    lang = rp.LANGUAGES[lang_name]
    sections = rp.patient_sections(rx_final, visit)
    cache_key = (lang, str(sections))
    if st.session_state.get("guide_key") != cache_key:
        with st.spinner(f"Translating to {lang_name}…"):
            st.session_state.guide, st.session_state.guide_warn = rp.translate_sections(sections, lang)
        st.session_state.guide_key = cache_key
        st.session_state.pop("audio", None)
    guide = st.session_state.guide
    for warn in st.session_state.guide_warn:
        st.warning(f"A number changed in translation, please check: “{warn['line']}” → “{warn['translated']}”")
    for (head, items), (_, en_items) in zip(guide, sections):
        st.markdown(f"#### {head}")
        for item, en in zip(items, en_items):
            st.markdown(f"- {item}" + (f"  \n  <small style='color:gray'>{html.escape(en)}</small>" if lang != "en-IN" else ""),
                        unsafe_allow_html=True)
    if st.button("🔊 Create voice note for the patient"):
        with st.spinner("Recording voice note with Bulbul…"):
            st.session_state.audio = rp.speak(f"{patient} जी, " * (lang == "hi-IN") + rp.sections_to_text(guide), lang)
    if st.session_state.get("audio"):
        st.audio(st.session_state.audio, format="audio/wav")
        st.download_button("Download voice note (send on WhatsApp)", st.session_state.audio,
                           file_name=f"parchi_{patient.replace(' ', '_')}_{lang}.wav", mime="audio/wav")
