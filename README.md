# 🩺 Parchi — the prescription that listens

Doctors in Indian OPDs see 60–100 patients a day and write prescriptions that patients can't read.
Diet advice, referrals and follow-up dates are said out loud and forgotten by the time the patient
reaches the pharmacy.

Parchi listens to the consultation (Hindi, English, or both mixed together) and produces:

1. **A complete, printable prescription for the doctor to sign.** It covers medicines with
   morning-afternoon-night dosing, before or after food, duration, diet restrictions, tests,
   referrals, the follow-up date and danger signs.
2. **A patient guide in the patient's own language**, as text and as a voice note that can be
   sent on WhatsApp.

## Why the output can be trusted

LLMs make dosing mistakes. In our first experiment, the model turned `1-0-1` (twice a day) into
"three times a day". So in Parchi **the model only extracts; deterministic code verifies.**

| Step | How it is done |
|---|---|
| Transcribe | Saaras v4 Batch, `codemix` mode, speaker diarization, plus the doctor's own medicine list as keyterms |
| Extract | Sarvam-105B returns structured JSON, and every medicine must quote the exact words it came from |
| **Verify** | Code checks that the medicine name and strength were actually said, that the quote matches the transcript, and that the timing (morning, afternoon, night, "twice a day") and food instruction agree with what the doctor said. Any mismatch is flagged in red. |
| Doctor review | Everything is editable, flags update live, and the doctor must approve before anything is produced |
| Patient text | Built from fixed templates (no LLM), so dose counts can't drift |
| Translate | Sarvam-Translate, line by line. Any line where a number changed during translation is flagged |
| Voice | Bulbul v3 voice note in 11 Indian languages |

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
echo "SARVAM_API_KEY=your_key_here" > .env
.venv/bin/streamlit run app.py
```

Click **▶️ Sample consultation → Use this sample → Write the Parchi** for an 83-second demo,
or record a live consultation in the browser.

To regenerate the sample audio: `.venv/bin/python samples/make_sample.py`.

## Deploy (Streamlit Community Cloud)

Push this folder to GitHub, create an app pointing at `app.py`, and add
`SARVAM_API_KEY = "..."` under **Settings → Secrets**. Never commit `.env`; `.gitignore`
already excludes it.

## Files

- `app.py`: Streamlit UI (record → review → prescription + patient guide)
- `rx_pipeline.py`: transcription, extraction, verification, templates, translation, speech
- `sarvam_client.py`: shared client that reads `SARVAM_API_KEY` from the environment or `.env`
- `samples/`: synthetic doctor-patient consultation and the script that creates it
