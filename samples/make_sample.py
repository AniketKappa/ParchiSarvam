"""Generate a synthetic doctor-patient consultation (code-mixed Hindi) for testing."""
import base64
import io
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sarvam_client import client  # noqa: E402

DOCTOR, PATIENT = "aditya", "ritu"

DIALOGUE = [
    (PATIENT, "नमस्ते डॉक्टर साहब। पिछले दो हफ्ते से बहुत थकान है, बार बार पेशाब आता है और प्यास भी बहुत लगती है।"),
    (DOCTOR, "अच्छा। आपकी sugar report देखी, fasting sugar एक सौ अस्सी है और BP भी एक सौ पचास बटा पचानवे है। ये diabetes और high BP है, घबराइए मत, control हो जाएगा।"),
    (DOCTOR, "Metformin पाँच सौ mg, सुबह और रात, खाने के बाद, एक महीने तक। Amlodipine पाँच mg, रात को एक गोली, एक महीने तक।"),
    (DOCTOR, "Pantoprazole चालीस mg, सुबह खाली पेट, नाश्ते से आधा घंटा पहले, पंद्रह दिन। और Vitamin D sixty thousand, हफ्ते में एक बार, रविवार को, आठ हफ्ते।"),
    (PATIENT, "खाने में क्या परहेज़ करना है डॉक्टर साहब?"),
    (DOCTOR, "चीनी, मिठाई, कोल्ड ड्रिंक बिल्कुल बंद। चावल कम कीजिए, रोटी चलेगी। नमक कम, अचार और पापड़ नहीं। रोज़ तीस मिनट walk कीजिए।"),
    (DOCTOR, "आँखों की जाँच के लिए eye specialist को दिखाइए, retina check करवाना है। और HbA1c, lipid profile और kidney function test करवा लीजिए।"),
    (DOCTOR, "दो हफ्ते बाद reports लेकर दोबारा आइए। अगर चक्कर आए, बहुत पसीना आए या सीने में दर्द हो तो तुरंत emergency में आइए।"),
    (PATIENT, "ठीक है डॉक्टर साहब, धन्यवाद।"),
]


def tts_frames(text: str, speaker: str) -> tuple[bytes, wave._wave_params]:
    resp = client.text_to_speech.convert(
        text=text, language_code="hi-IN", speaker=speaker, model="bulbul:v3", speech_sample_rate=16000,
    )
    with wave.open(io.BytesIO(base64.b64decode("".join(resp.audios)))) as w:
        return w.readframes(w.getnframes()), w.getparams()


def main() -> None:
    out = Path(__file__).with_name("consultation_hi.wav")
    params, chunks = None, []
    for speaker, line in DIALOGUE:
        frames, params = tts_frames(line, speaker)
        chunks.append(frames)
        chunks.append(b"\x00\x00" * int(params.framerate * 0.6))  # pause between turns
    with wave.open(str(out), "wb") as w:
        w.setparams(params)
        w.writeframes(b"".join(chunks))
    with wave.open(str(out)) as w:
        print(f"Wrote {out.name}: {w.getnframes() / w.getframerate():.1f}s")


if __name__ == "__main__":
    main()
