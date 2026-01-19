import os
import joblib
import pandas as pd
import streamlit as st
import numpy as np

# KONFIG
MODEL_PATH = os.getenv("MODEL_PATH", "models/catboost_price.joblib")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "models/feature_schema.joblib")


def fmt_pln(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


st.set_page_config(page_title="Profesjonalna wycena samochodu", layout="wide")
st.markdown("""
<style>
:root{
  --bg: #F6F7FB;
  --card: #FFFFFF;
  --text: #0F172A;
  --muted: #475569;
  --border: #D7DDEA;

  --blue: #004D98;
  --maroon: #A50044;
  --gold: #D4AF37;

  --shadow: 0 2px 10px rgba(15, 23, 42, 0.08);
}

.stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1100px; }

h1, h2, h3 { color: var(--text); }
.subtitle { color: var(--muted); font-size: 0.98rem; margin-bottom: 1.2rem; }

/* Sekcje jako karty */
.section{
  padding: 0.95rem 1.0rem;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: var(--card);
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
  margin-bottom: 0.9rem;
}

/* Tytuł sekcji z akcentem */
.section-title{
  font-weight: 900;
  margin-bottom: 0.6rem;
  font-size: 1.02rem;
  color: var(--blue);
}
.section-title::after{
  display: none;
  content: none;
}


/* Wynik */
.result-card{
  padding: 1.0rem 1.0rem;
  border-radius: 18px;
  border: 1px solid var(--border);
  background: var(--card);
  box-shadow: var(--shadow);
  margin-top: 1.0rem;
}
.result-label{ color: var(--muted); font-size: 0.95rem; margin-bottom: 0.25rem; }
.result-range{
  font-size: 1.85rem;
  font-weight: 950;
  line-height: 1.15;
  color: var(--maroon);
}
.result-note{ color: var(--muted); font-size: 0.92rem; margin-top: 0.5rem; }

a, a:visited { color: var(--blue); }
hr { border-color: var(--border); }

/* Przycisk primary */
button[kind="primary"]{
  background: linear-gradient(90deg, var(--blue), var(--maroon)) !important;
  color: white !important;
  border-radius: 12px !important;
  padding: 0.65rem 0.95rem !important;
  font-weight: 900 !important;
  border: 1px solid rgba(0,0,0,0) !important;
  box-shadow: 0 6px 18px rgba(0, 77, 152, 0.20);
}
button[kind="primary"]:hover{
  filter: brightness(0.97);
}

button[kind="secondary"]{
  border-radius: 12px !important;
  padding: 0.65rem 0.95rem !important;
  font-weight: 800 !important;
}

/* Streamlit elementy */
div[data-testid="stForm"] { border: none; padding: 0; }
div[data-testid="stHorizontalBlock"] { gap: 0.8rem; }

/* Poprawa czytelności captionów */
.stCaption { color: var(--muted) !important; }

.gold-badge{
  display:inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  border: 1px solid rgba(212, 175, 55, 0.55);
  background: rgba(212, 175, 55, 0.10);
  color: #6B4E00;
  font-weight: 800;
  font-size: 0.85rem;
}

/* Kontenery Streamlit z border=True */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 0.95rem 1rem !important;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05) !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Wycena samochodu (ML)")
st.markdown('<div class="subtitle">Estymacja cen samochodu na podstawie danych z otomoto.pl (2021). Wynik prezentowany jako widełki +-10%.</div>', unsafe_allow_html=True)

CATEGORICAL_OPTIONS = {
    "Offer_location": [
        "dolnośląskie", "kujawsko-pomorskie", "lubelskie", "lubuskie",
        "małopolskie", "mazowieckie", "opolskie", "podkarpackie",
        "podlaskie", "pomorskie", "śląskie", "świętokrzyskie",
        "warmińsko-mazurskie", "wielkopolskie", "łódzkie",
        "Brak danych",
    ],
    "Fuel_type": [
        "Diesel", "Electric", "Ethanol", "Gasoline", "Gasoline + LPG",
        "Hybrid", "LPG", "Natural Gas", "Brak danych",
    ],
    "Drive": [
        "Front wheels", "4x4 (permanent)", "4x4 (automatic)",
        "4x4 (connected manually)", "Rear wheels", "Brak danych",
    ],
    "Transmission": ["Automatic", "Manual"],
    "Condition": ["New", "Used"],
    "Type": [
        "SUV", "Sedan", "Hatchback", "Station wagon", "Coupe",
        "Van", "Convertible", "Pickup", "Small car", "Brak danych",
    ],
    "Colour": [
        "Black", "White", "Gray", "Silver", "Blue", "Red", "Brown", "Green",
        "Beige", "Gold", "Orange", "Yellow", "Purple", "Brak danych",
    ],
    "Origin_country": [
        "Germany", "Poland", "Brak danych", "France", "Other",
        "Belgium", "Netherlands",
    ],
    "First_owner": ["Yes", "Brak danych"],
}

TOP_BRANDS = [
    "Volkswagen", "BMW", "Audi", "Opel", "Ford", "Mercedes-Benz", "Renault",
    "Toyota", "Škoda", "Peugeot", "Citroën", "Volvo", "Kia", "Hyundai",
    "Fiat", "Seat", "Nissan", "Mazda", "Honda", "Suzuki",
    "Mitsubishi", "Jeep", "Dacia", "Chevrolet", "MINI", "Alfa Romeo",
    "Land Rover", "Porsche", "Jaguar", "Lexus", "Subaru", "Chrysler",
    "Dodge", "Saab", "Smart", "Infiniti", "Lancia", "SsangYong",
    "Maserati", "Cadillac",
]

BRAND_TO_MODELS = {
    "Volkswagen": [
        "Golf", "Passat", "Polo", "Tiguan", "Touran", "Caddy", "Golf Plus",
        "Arteon", "Sharan", "up!", "Jetta", "Transporter", "Touareg",
        "T-Roc", "T-Cross", "Multivan", "Caravelle", "CC", "Scirocco",
        "New Beetle", "Golf Sportsvan", "Passat CC", "Fox", "Bora", "Beetle",
    ],
    "BMW": [
        "Seria 3", "Seria 5", "Seria 1", "X3", "X5", "Seria 7", "X1",
        "Seria 2", "Seria 4", "X6", "i3", "3GT", "Seria 6", "X4", "X2",
        "M5", "X5 M", "M3", "X7", "Seria 8", "X6 M", "5GT", "M4", "M8", "Z4",
    ],
    "Audi": [
        "A4", "A6", "A3", "Q5", "A5", "A8", "Q7", "Q3", "A7", "A6 Allroad",
        "A1", "A4 Allroad", "Q2", "TT", "S3", "Q8", "S5", "RS6", "SQ7",
        "S6", "RS Q3", "S4", "SQ5", "A2", "S8",
    ],
    "Opel": [
        "Astra", "Insignia", "Corsa", "Zafira", "Meriva", "Vectra", "Mokka",
        "Vivaro", "Crossland X", "Grandland X", "Combo", "Signum", "Antara",
        "Agila", "Tigra", "Adam", "Karl", "Omega", "Movano", "Frontera",
        "Other", "Ampera", "Cascada", "Kadett", "Rekord",
    ],
    "Ford": [
        "Focus", "Mondeo", "Fiesta", "S-Max", "Kuga", "C-MAX", "Mustang",
        "Galaxy", "Fusion", "EcoSport", "Puma", "Grand C-MAX", "EDGE", "KA",
        "Tourneo Custom", "B-MAX", "Ranger", "Transit", "Focus C-Max",
        "Transit Custom", "Escape", "Explorer", "Tourneo Connect",
        "Tourneo Courier", "Transit Connect",
    ],
    "Mercedes-Benz": [
        "Klasa E", "Klasa C", "Klasa A", "Klasa S", "Klasa B", "CLA", "GLE",
        "GLC", "CLS", "ML", "GLA", "SL", "Vito", "CLK",
        "W124 (1984-1993)", "Other", "GL", "CL", "GLK", "Klasa V", "GLS",
        "SLK", "Sprinter", "AMG GT", "Viano",
    ],
    "Renault": [
        "Megane", "Clio", "Scenic", "Laguna", "Captur", "Grand Scenic",
        "Kadjar", "Trafic", "Espace", "Talisman", "Twingo", "Modus",
        "Kangoo", "Koleos", "Grand Espace", "Fluence", "Zoe", "Thalia",
        "Master", "Latitude", "Vel Satis", "Other", "Scenic Conquest",
        "Safrane", "19",
    ],
    "Toyota": [
        "Avensis", "Yaris", "Corolla", "Auris", "RAV4", "C-HR", "Aygo",
        "Verso", "Corolla Verso", "Prius", "Camry", "Land Cruiser",
        "Proace City Verso", "Proace Verso", "Sienna", "Hilux", "ProAce",
        "Verso S", "Highlander", "Celica", "Prius+", "Tundra", "Yaris Verso",
        "Avensis Verso", "Supra",
    ],
    "Škoda": [
        "Octavia", "Fabia", "Superb", "RAPID", "Scala", "Citigo", "Kodiaq",
        "Karoq", "Roomster", "Yeti", "Kamiq", "Enyaq", "Felicia", "Praktik",
        "105", "Favorit", "Forman", "130",
    ],
    "Peugeot": [
        "308", "508", "3008", "208", "207", "5008", "2008", "407", "307",
        "Partner", "206", "Rifter", "301", "107", "Expert", "307 CC", "807",
        "207 CC", "206 plus", "Boxer", "1007", "206 CC", "RCZ", "607",
        "Traveller",
    ],
    "Citroën": [
        "C5", "C3", "C4", "C4 Picasso", "Berlingo", "C4 Grand Picasso",
        "C5 Aircross", "C1", "C3 Picasso", "C3 Aircross", "Xsara Picasso",
        "DS3", "C4 Cactus", "C2", "DS4", "DS5", "C-Elysée", "Jumpy Combi",
        "C8", "Xsara", "Other", "SpaceTourer", "Jumper", "C4 Aircross",
        "C-Crosser",
    ],
    "Volvo": [
        "XC 60", "V40", "V60", "V50", "S60", "XC 90", "V70", "V90", "S80",
        "S40", "XC 40", "S90", "C30", "XC 70", "C70", "Other", "850",
        "Seria 900", "S70", "Seria 400", "340", "Seria 700", "262", "965",
        "744",
    ],
    "Kia": [
        "Ceed", "Sportage", "Picanto", "Rio", "Venga", "Optima", "Stonic",
        "Sorento", "XCeed", "Carens", "Pro_cee'd", "Soul", "Niro", "Stinger",
        "Carnival", "Magentis", "Other", "Cerato", "Opirus", "Shuma", "Joice",
        "Sedona", "Clarus",
    ],
    "Hyundai": [
        "I30", "Tucson", "ix35", "i20", "i40", "Kona", "Santa Fe", "i10",
        "ix20", "Elantra", "Getz", "IONIQ", "Coupe", "Sonata", "i30 N",
        "Veloster", "Matrix", "Accent", "Genesis Coupe", "Atos", "Terracan",
        "H-1", "Galloper", "Other", "Genesis",
    ],
    "Fiat": [
        "Tipo", "500", "Panda", "Grande Punto", "Bravo", "Punto", "Doblo",
        "500X", "Punto Evo", "Freemont", "500L", "Sedici", "Croma", "Stilo",
        "Ducato", "126", "Seicento", "Scudo", "Linea", "Fiorino", "Qubo",
        "Punto 2012", "Cinquecento", "125p", "Talento",
    ],
    "Seat": [
        "Leon", "Ibiza", "Altea", "Arona", "Altea XL", "Alhambra", "Toledo",
        "Ateca", "Exeo", "Tarraco", "Mii", "Cordoba", "Arosa", "Marbella",
        "Ronda",
    ],
    "Nissan": [
        "Qashqai", "Juke", "Micra", "X-Trail", "Note", "Qashqai+2", "Leaf",
        "Primera", "Almera", "Patrol", "Murano", "Navara", "Pulsar",
        "Pathfinder", "Tiida", "Almera Tino", "NV200", "Pixo", "370 Z",
        "Primastar", "GT-R", "Altima", "350 Z", "300 ZX", "Terrano",
    ],
    "Mazda": [
        "6", "3", "CX-5", "5", "2", "CX-3", "CX-30", "MX-5", "CX-7", "CX-9",
        "Premacy", "RX-8", "MX-30", "323F", "323", "Tribute", "626", "MPV",
        "MX-3", "RX-7", "Demio", "121", "Other", "BT-50", "Xedos",
    ],
    "Honda": [
        "Civic", "CR-V", "Accord", "Jazz", "HR-V", "FR-V", "Odyssey",
        "Legend", "City", "Other", "Insight", "Prelude", "CR-Z", "CRX",
        "Stream", "S 2000", "Pilot", "Ridgeline", "Integra",
    ],
    "Suzuki": [
        "Swift", "Vitara", "SX4", "SX4 S-Cross", "Grand Vitara", "Ignis",
        "Jimny", "Baleno", "Splash", "Alto", "Swace", "Liana", "Celerio",
        "Samurai", "Wagon R+", "Across", "Kizashi", "Other", "X-90", "XL7",
        "LJ", "SJ",
    ],
}


# ŁADOWANIE MODELU + SCHEMATU
@st.cache_resource
def load_model_and_schema():
    model = joblib.load(MODEL_PATH)
    schema = joblib.load(SCHEMA_PATH)
    return model, schema


def build_features_row(user_input: dict, schema: dict) -> pd.DataFrame:
    cols = schema["feature_columns"]
    cat_cols = set(schema.get("cat_cols", []))
    num_cols = set(schema.get("num_cols", []))

    row = {}
    for c in cols:
        if c in user_input:
            row[c] = user_input[c]
        else:
            row[c] = "Brak danych" if c in cat_cols else 0

    X_one = pd.DataFrame([row], columns=cols)

    for c in cat_cols:
        if c in X_one.columns:
            X_one[c] = X_one[c].astype(str).fillna("Brak danych")

    for c in num_cols:
        if c in X_one.columns:
            X_one[c] = pd.to_numeric(X_one[c], errors="coerce").fillna(0)

    return X_one


def clean_choice(v: str) -> str:
    return "Brak danych" if v in (None, "", "[nie wybrano]") else v


def select_or_manual(label: str, options: list[str], key: str, default_index: int = 0):
    if not options:
        return st.text_input(label, key=f"{key}_manual")

    opts = options + ["Inne (wpisz ręcznie)"]
    picked = st.selectbox(label, opts, index=min(default_index, len(opts) - 1), key=key)
    if picked == "Inne (wpisz ręcznie)":
        return st.text_input("Wpisz wartość:", key=f"{key}_manual")
    return picked

def clear_model_state():
    for k in list(st.session_state.keys()):
        if k.startswith("model_") or k.startswith("model_manual"):
            del st.session_state[k]


# START
try:
    model, schema = load_model_and_schema()
except FileNotFoundError:
    st.error(
        f"Brak plików modelu.\n\n"
        f"- {MODEL_PATH}\n"
        f"- {SCHEMA_PATH}\n\n"
        f"Najpierw wytrenuj model i wrzuć pliki do katalogu models/."
    )
    st.stop()
st.markdown("## Formularz wyceny")

# PODSTAWOWE INFORMACJE
with st.container(border=True):
    st.markdown('<div class="section-title">Podstawowe informacje</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)

    with a1:
        brand_opts = TOP_BRANDS + ["Inne (wpisz ręcznie)"]
        picked_brand = st.selectbox(
            "Marka (Vehicle_brand)",
            brand_opts,
            key="brand",
            index=0,
            on_change=clear_model_state,
        )
        if picked_brand == "Inne (wpisz ręcznie)":
            brand = st.text_input("Wpisz markę:", key="brand_manual", on_change=clear_model_state)
        else:
            brand = picked_brand

    with a2:
        model_options = BRAND_TO_MODELS.get(brand, [])

        if model_options:
            model_opts = model_options + ["Inne (wpisz ręcznie)"]
            picked_model = st.selectbox(
                "Model (Vehicle_model)",
                model_opts,
                key=f"model_{brand}",
                index=0
            )
            if picked_model == "Inne (wpisz ręcznie)":
                vehicle_model = st.text_input("Wpisz model:", key=f"model_manual_{brand}")
            else:
                vehicle_model = picked_model
        else:
            vehicle_model = st.text_input("Model (Vehicle_model)", key=f"model_manual_{brand}")

    with a3:
        car_type = st.selectbox(
            "Typ nadwozia (Type)",
            CATEGORICAL_OPTIONS["Type"] + ["Inne (wpisz ręcznie)"],
            key="type",
            index=0
        )
        if car_type == "Inne (wpisz ręcznie)":
            car_type = st.text_input("Wpisz typ:", key="type_manual")

    st.write("")  # odstęp
with st.form("valuation_form"):
    # PARAMETRY LICZBOWE
    with st.container(border=True):
        st.markdown('<div class="section-title">Parametry liczbowe</div>', unsafe_allow_html=True)

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            year = st.number_input("Rok produkcji", min_value=1915, max_value=2021, value=2006, step=1)
        with b2:
            mileage = st.number_input("Przebieg [km]", min_value=0, max_value=2_000_000, value=200_000, step=1000)
        with b3:
            power = st.number_input("Moc [KM]", min_value=0, max_value=1200, value=120, step=10)
        with b4:
            displacement = st.number_input("Pojemność [cm³]", min_value=0, max_value=10_000, value=1600, step=100)

    st.write("")

    # DODATKOWE DANE
    with st.container(border=True):
        st.markdown('<div class="section-title">Dodatkowe dane</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            fuel = select_or_manual("Paliwo (Fuel_type)", CATEGORICAL_OPTIONS["Fuel_type"], key="fuel", default_index=0)
            drive = select_or_manual("Napęd (Drive)", CATEGORICAL_OPTIONS["Drive"], key="drive", default_index=0)

        with c2:
            transmission = select_or_manual("Skrzynia biegów (Transmission)", CATEGORICAL_OPTIONS["Transmission"], key="trans", default_index=0)
            doors_number = st.number_input("Liczba drzwi (Doors_number)", min_value=0, max_value=10, value=5, step=1)

        with c3:
            colour = select_or_manual("Kolor (Colour)", CATEGORICAL_OPTIONS["Colour"], key="colour", default_index=0)
            condition = select_or_manual("Stan (Condition)", CATEGORICAL_OPTIONS["Condition"], key="cond", default_index=1)

    st.write("")

    # POCHODZENIE I LOKALIZACJA
    with st.container(border=True):
        st.markdown('<div class="section-title">Pochodzenie i lokalizacja</div>', unsafe_allow_html=True)

        d1, d2, d3 = st.columns(3)

        with d1:
            origin_country = select_or_manual(
                "Kraj pochodzenia (Origin_country)",
                CATEGORICAL_OPTIONS["Origin_country"],
                key="origin",
                default_index=0
            )
            first_owner = select_or_manual(
                "Pierwszy właściciel (First_owner)",
                CATEGORICAL_OPTIONS["First_owner"],
                key="first_owner",
                default_index=0
            )

        with d2:
            location = select_or_manual(
                "Województwo (Offer_location)",
                CATEGORICAL_OPTIONS["Offer_location"],
                key="loc",
                default_index=0
            )

        with d3:
            st.caption("Cena nie jest potrzebna do wyceny (to target w danych treningowych).")
            note = st.text_input("Uwagi (opcjonalnie)", value="")

    st.write("")

    # SUBMIT
    submit_col1, submit_col2 = st.columns([1, 2])
    with submit_col1:
        submitted = st.form_submit_button("Wyceń", type="primary")
    with submit_col2:
        st.caption("Wynik pokazujemy jako widełki ±10% od estymacji modelu.")

# WYNIK
if submitted:
    user_input = {
        "Condition": clean_choice(condition),
        "Vehicle_brand": clean_choice(brand),
        "Vehicle_model": clean_choice(vehicle_model),
        "Production_year": int(year),
        "Mileage_km": int(mileage),
        "Power_HP": float(power),
        "Displacement_cm3": float(displacement),
        "Fuel_type": clean_choice(fuel),
        "Drive": clean_choice(drive),
        "Transmission": clean_choice(transmission),
        "Type": clean_choice(car_type),
        "Doors_number": int(doors_number),
        "Colour": clean_choice(colour),
        "Origin_country": clean_choice(origin_country),
        "First_owner": clean_choice(first_owner),
        "Offer_location": clean_choice(location),
    }

    with st.spinner("Liczymy wycenę..."):
        X_one = build_features_row(user_input, schema)
        pred = float(model.predict(X_one)[0])

        if schema.get("use_log_target", False):
            pred = float(np.expm1(pred))

    low = pred * 0.9
    high = pred * 1.1

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Szacowany przedział ceny (±10%)</div>
            <div class="result-range">{fmt_pln(low)} – {fmt_pln(high)} PLN</div>
            <div class="result-note">
                To estymacja na podstawie ogłoszeń z 2021 roku. Realna cena zależy m.in. od stanu, wersji wyposażenia, historii serwisowej i popytu lokalnego.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    tabs = st.tabs(["Podsumowanie danych", "Wiersz cech do modelu", "Notatka"])
    with tabs[0]:
        st.json(user_input)
    with tabs[1]:
        st.dataframe(X_one, use_container_width=True)
    with tabs[2]:
        st.write(note if note else "Brak uwag.")
