import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
import base64

# ── Seiteneinstellungen ────────────────────────────────────────────────────────
st.set_page_config(page_title="Fundbüro", page_icon="🎒", layout="centered")

# ── Logo laden (logo.png im selben Ordner) ────────────────────────────────────
def logo_laden():
    logo_pfad = "logo.png"
    if os.path.exists(logo_pfad):
        with open(logo_pfad, "rb") as f:
            daten = f.read()
        b64 = base64.b64encode(daten).decode()
        return f'<img src="data:image/png;base64,{b64}" style="height:70px; display:block; margin-left:auto;">'
    else:
        return (
            '<div style="text-align:right; font-family:\'IM Fell English\',\'Palatino Linotype\',serif;'
            'font-size:13px; font-weight:bold; color:red; line-height:1.2; font-style:italic;">'
            'Katharineum<br>zu Lübeck<br>'
            '<span style="color:black; font-size:11px; font-style:italic;">TU ES</span>'
            '</div>'
        )

LOGO_HTML = logo_laden()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IM+Fell+English:ital@0;1&display=swap');

/* Hintergrund weiß */
.stApp { background-color: white; }

/* Abstände */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 480px !important;
}

/* ── ALLE Buttons generell: IM Fell English kursiv ── */
button, .stButton > button {
    font-family: 'IM Fell English', 'Palatino Linotype', serif !important;
    font-style: italic !important;
}

/* ── Startseiten-Buttons: halbe Bildschirmhöhe ── */
.grosser-button > button {
    font-family: 'IM Fell English', 'Palatino Linotype', serif !important;
    font-size: 32px !important;
    font-style: italic !important;
    width: 100% !important;
    min-height: 45vh !important;
    background-color: white !important;
    color: black !important;
    border: 2.5px solid black !important;
    border-radius: 4px !important;
    margin-bottom: 18px !important;
    text-align: left !important;
    padding-left: 32px !important;
    display: flex !important;
    align-items: center !important;
}
.grosser-button > button:hover {
    background-color: #f5f5f5 !important;
}

/* ── Fertig-Button türkis ── */
.fertig-button > button {
    font-family: 'IM Fell English', 'Palatino Linotype', serif !important;
    font-size: 22px !important;
    font-style: italic !important;
    background-color: #3dd6b5 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    width: 100% !important;
    height: 60px !important;
    margin-top: 10px !important;
}
.fertig-button > button:hover {
    background-color: #2bbfa0 !important;
}

/* ── Zurück-Button ── */
.zurueck-button > button {
    font-family: 'IM Fell English', 'Palatino Linotype', serif !important;
    font-size: 18px !important;
    font-style: italic !important;
    background-color: white !important;
    color: black !important;
    border: 2px solid black !important;
    border-radius: 4px !important;
    width: 100% !important;
    height: 50px !important;
    margin-top: 12px !important;
}

/* ── Eingabefelder ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    font-family: 'IM Fell English', 'Palatino Linotype', serif !important;
    font-style: italic !important;
    font-size: 20px !important;
    border: 2.5px solid black !important;
    border-radius: 4px !important;
    background-color: white !important;
    padding: 18px 16px !important;
    height: 70px !important;
}
.stTextArea > div > div > textarea {
    height: 90px !important;
}

/* ── File-Uploader ── */
.stFileUploader > div {
    border: 2.5px solid black !important;
    border-radius: 4px !important;
    background-color: white !important;
    padding: 10px !important;
}
.stFileUploader label,
.stFileUploader span,
.stFileUploader p {
    font-family: 'IM Fell English', 'Palatino Linotype', serif !important;
    font-style: italic !important;
    font-size: 20px !important;
}

/* ── Fundstück-Karte ── */
.karte-name {
    font-family: 'IM Fell English', 'Palatino Linotype', serif;
    font-size: 19px;
    font-style: italic;
    font-weight: bold;
    text-decoration: underline;
    margin-bottom: 4px;
}
.karte-beschreibung {
    font-family: 'IM Fell English', 'Palatino Linotype', serif;
    font-size: 14px;
    font-style: italic;
    color: #222;
}

/* ── Startseiten-Titel ── */
.start-titel {
    font-family: 'IM Fell English', 'Palatino Linotype', serif;
    font-size: 28px;
    font-style: italic;
    font-weight: bold;
    color: black;
    text-align: center;
    margin-bottom: 20px;
}

/* ── Labels & Streamlit-Elemente ausblenden ── */
.stTextInput label, .stTextArea label { display: none !important; }
hr { display: none; }
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stVerticalBlock"] > div { margin-bottom: 0px !important; }
</style>
""", unsafe_allow_html=True)

# ── Modell laden ───────────────────────────────────────────────────────────────
@st.cache_resource
def modell_laden():
    model = load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

model, labels = modell_laden()

# ── KI: Bild erkennen ─────────────────────────────────────────────────────────
def gegenstand_erkennen(bild: Image.Image):
    bild = bild.convert("RGB").resize((224, 224))
    bild_array = np.asarray(bild, dtype=np.float32)
    bild_array = (bild_array / 127.5) - 1
    bild_array = np.expand_dims(bild_array, axis=0)
    vorhersage = model.predict(bild_array)
    index = np.argmax(vorhersage)
    return labels[index], float(vorhersage[0][index])

# ── Session State ──────────────────────────────────────────────────────────────
if "seite" not in st.session_state:
    st.session_state.seite = "start"
if "fundstuecke" not in st.session_state:
    st.session_state.fundstuecke = []

# ══════════════════════════════════════════════════════════════════════════════
# STARTSEITE
# ══════════════════════════════════════════════════════════════════════════════
def startseite():
    st.markdown(LOGO_HTML, unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="start-titel">Fundbüro 🎒</div>', unsafe_allow_html=True)

    # Button: Suchen (halbe Bildschirmhöhe)
    st.markdown('<div class="grosser-button">', unsafe_allow_html=True)
    if st.button("Suchen 🔍", key="btn_suchen"):
        st.session_state.seite = "suchen"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Button: Hochladen (halbe Bildschirmhöhe)
    st.markdown('<div class="grosser-button">', unsafe_allow_html=True)
    if st.button("Hochladen ⬆", key="btn_hochladen"):
        st.session_state.seite = "hochladen"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SUCHSEITE
# ══════════════════════════════════════════════════════════════════════════════
def suchseite():
    suchbegriff = st.text_input("", placeholder="Suchen 🔍", key="suchfeld")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    gefundene = (
        [
            f for f in st.session_state.fundstuecke
            if suchbegriff.lower() in f["name"].lower()
            or suchbegriff.lower() in f["beschreibung"].lower()
        ]
        if suchbegriff
        else st.session_state.fundstuecke
    )

    if not gefundene:
        st.markdown(
            "<p style='font-family:IM Fell English,Palatino Linotype,serif;"
            "font-style:italic;color:gray;text-align:center;margin-top:20px;'>"
            "Keine Fundstücke gefunden.</p>",
            unsafe_allow_html=True
        )
    else:
        for fund in gefundene:
            col_bild, col_text = st.columns([1, 3])
            with col_bild:
                if fund["bild"] is not None:
                    st.image(fund["bild"], width=75)
                else:
                    st.markdown(
                        "<div style='font-size:40px;text-align:center;'>🖼</div>",
                        unsafe_allow_html=True
                    )
            with col_text:
                st.markdown(
                    f'<div class="karte-name">{fund["name"]}</div>'
                    f'<div class="karte-beschreibung">Beschreibung:<br>{fund["beschreibung"]}</div>',
                    unsafe_allow_html=True
                )
            st.markdown(
                "<div style='border-top:2px solid black;margin-bottom:12px;'></div>",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="zurueck-button">', unsafe_allow_html=True)
    if st.button("← Zurück", key="suche_zurueck"):
        st.session_state.seite = "start"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HOCHLADESEITE
# ══════════════════════════════════════════════════════════════════════════════
def hochladeseite():
    st.markdown(LOGO_HTML, unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    hochgeladenes_bild = st.file_uploader("Foto hochladen ⬆", type=["jpg", "jpeg", "png"])
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    name = st.text_input("", placeholder="Name des Objekts", key="name_input")
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    beschreibung = st.text_area("", placeholder="Beschreibung", key="beschreibung_input", height=90)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="fertig-button">', unsafe_allow_html=True)
    fertig = st.button("Fertig", key="fertig_btn")
    st.markdown('</div>', unsafe_allow_html=True)

    if fertig:
        bild_objekt = None
        ki_name = name
        if hochgeladenes_bild is not None:
            bild_objekt = Image.open(hochgeladenes_bild)
            if not name.strip():
                ki_name, konfidenz = gegenstand_erkennen(bild_objekt)
        if ki_name.strip():
            st.session_state.fundstuecke.append({
                "name": ki_name,
                "beschreibung": beschreibung,
                "bild": bild_objekt
            })
            st.success(f"✅ '{ki_name}' wurde gespeichert!")
            st.session_state.seite = "start"
            st.rerun()
        else:
            st.warning("Bitte gib einen Namen ein oder lade ein Foto hoch.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="zurueck-button">', unsafe_allow_html=True)
    if st.button("← Zurück", key="hochlade_zurueck"):
        st.session_state.seite = "start"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEITENSTEUERUNG
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.seite == "start":
    startseite()
elif st.session_state.seite == "suchen":
    suchseite()
elif st.session_state.seite == "hochladen":
    hochladeseite()
