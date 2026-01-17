import os
import joblib
import pandas as pd
import streamlit as st

# =========================
# KONFIG
# =========================
MODEL_PATH = os.getenv("MODEL_PATH", "models/catboost_price.joblib")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "models/feature_schema.joblib")

st.set_page_config(page_title="Wycena samochodu (ML)", layout="wide")
st.title("Wycena samochodu (ML)")

st.write(
    """
Wpisujesz dane auta, klikasz i dostajesz wycenę.
To estymacja na podstawie danych ogłoszeniowych (model regresyjny), a nie wróżenie z fusów.
"""
)

# =========================
# OPCJA B: listy kategorii w kodzie (bez CSV)
# Wartości poniżej są dopasowane do tego, co występuje w Car_sale_ads_cleaned_v2.csv
# =========================

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

# Top marki z danych (żeby user nie musiał wpisywać „Citroën” z palca)
TOP_BRANDS = [
    "Volkswagen", "BMW", "Audi", "Opel", "Ford", "Mercedes-Benz", "Renault",
    "Toyota", "Škoda", "Peugeot", "Citroën", "Volvo", "Kia", "Hyundai",
    "Fiat", "Seat", "Nissan", "Mazda", "Honda", "Suzuki",
    "Mitsubishi", "Jeep", "Dacia", "Chevrolet", "MINI", "Alfa Romeo",
    "Land Rover", "Porsche", "Jaguar", "Lexus", "Subaru", "Chrysler",
    "Dodge", "Saab", "Smart", "Infiniti", "Lancia", "SsangYong",
    "Maserati", "Cadillac",
]

# Top modele per top marka (z danych). Jeśli marka nie jest na liście -> wpis ręczny.
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


# =========================
# ŁADOWANIE MODELU + SCHEMATU
# =========================
@st.cache_resource
def load_model_and_schema():
    model = joblib.load(MODEL_PATH)
    schema = joblib.load(SCHEMA_PATH)
    return model, schema


def build_features_row(user_input: dict, schema: dict) -> pd.DataFrame:
    """
    Buduje jeden wiersz cech w kolejności i typach jak w treningu.
    """
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
    """
    Selectbox + opcja 'Inne (wpisz ręcznie)'.
    Zwraca finalny string.
    """
    if not options:
        return st.text_input(label, key=f"{key}_manual")

    opts = options + ["Inne (wpisz ręcznie)"]
    picked = st.selectbox(label, opts, index=min(default_index, len(opts) - 1), key=key)
    if picked == "Inne (wpisz ręcznie)":
        return st.text_input("Wpisz wartość:", key=f"{key}_manual")
    return picked


# =========================
# START
# =========================
model_ok = True
try:
    model, schema = load_model_and_schema()
except FileNotFoundError:
    model_ok = False
    st.error(
        f"Brak plików modelu.\n\n"
        f"- {MODEL_PATH}\n"
        f"- {SCHEMA_PATH}\n\n"
        f"Najpierw wytrenuj model (train_model.py) i wrzuć pliki do katalogu models/."
    )

if not model_ok:
    st.stop()

st.markdown("## Formularz wyceny")

c1, c2, c3 = st.columns(3)

with c1:
    brand = select_or_manual("Marka (Vehicle_brand):", TOP_BRANDS, key="brand", default_index=0)

with c2:
    model_options = BRAND_TO_MODELS.get(brand, [])
    vehicle_model = select_or_manual("Model (Vehicle_model):", model_options, key="model", default_index=0)

with c3:
    car_type = select_or_manual("Typ nadwozia (Type):", CATEGORICAL_OPTIONS["Type"], key="type", default_index=0)

st.markdown("---")

with st.form("valuation_form"):
    left, right = st.columns(2)

    with left:
        year = st.number_input("Rok produkcji (Production_year):", min_value=1950, max_value=2030, value=2015, step=1)
        mileage = st.number_input("Przebieg [km] (Mileage_km):", min_value=0, max_value=2_000_000, value=150_000, step=1000)

    with right:
        power = st.number_input("Moc [KM] (Power_HP):", min_value=0, max_value=1200, value=120, step=10)
        displacement = st.number_input("Pojemność [cm³] (Displacement_cm3):", min_value=0, max_value=10_000, value=1600, step=100)

    c4, c5, c6 = st.columns(3)

    with c4:
        fuel = select_or_manual("Rodzaj paliwa (Fuel_type):", CATEGORICAL_OPTIONS["Fuel_type"], key="fuel", default_index=0)
        drive = select_or_manual("Napęd (Drive):", CATEGORICAL_OPTIONS["Drive"], key="drive", default_index=0)

    with c5:
        transmission = select_or_manual("Skrzynia biegów (Transmission):", CATEGORICAL_OPTIONS["Transmission"], key="trans", default_index=0)
        doors_number = st.number_input("Liczba drzwi (Doors_number):", min_value=0, max_value=10, value=5, step=1)

    with c6:
        colour = select_or_manual("Kolor (Colour):", CATEGORICAL_OPTIONS["Colour"], key="colour", default_index=0)
        condition = select_or_manual("Stan (Condition):", CATEGORICAL_OPTIONS["Condition"], key="cond", default_index=1)  # default Used

    c7, c8, c9 = st.columns(3)

    with c7:
        origin_country = select_or_manual("Kraj pochodzenia (Origin_country):", CATEGORICAL_OPTIONS["Origin_country"], key="origin", default_index=0)
        first_owner = select_or_manual("Pierwszy właściciel (First_owner):", CATEGORICAL_OPTIONS["First_owner"], key="first_owner", default_index=0)

    with c8:
        location = select_or_manual("Województwo (Offer_location):", CATEGORICAL_OPTIONS["Offer_location"], key="loc", default_index=0)

    with c9:
        st.caption("Cena nie jest potrzebna do wyceny. To target w danych treningowych.")
        _ = st.text_input("Uwagi (opcjonalnie, nie wpływa na model):", value="")

    submitted = st.form_submit_button("Wyceń")

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

    X_one = build_features_row(user_input, schema)
    pred = float(model.predict(X_one)[0])

    st.success("Wycena gotowa.")
    st.metric("Szacowana cena [PLN]", f"{pred:,.0f}".replace(",", " "))

    with st.expander("Pokaż dane wejściowe"):
        st.json(user_input)

    with st.expander("Pokaż wiersz cech przekazany do modelu"):
        st.dataframe(X_one)
