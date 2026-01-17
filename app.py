import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wycena samochodu", layout="wide")
st.title("Wycena samochodu (ML)")

st.write(
    "Wersja serwerowa: tylko formularz wyceny. Bez wczytywania ogromnych CSV, "
    "bo Render na 512MB RAM nie jest miejscem na marzenia o big data."
)

MODEL_PATH = "models/catboost_price.joblib"
SCHEMA_PATH = "models/feature_schema.joblib"
UI_META_PATH = "models/ui_metadata.joblib"  # opcjonalne, ale polecane


@st.cache_resource
def load_model_and_schema():
    model = joblib.load(MODEL_PATH)
    schema = joblib.load(SCHEMA_PATH)
    return model, schema


@st.cache_resource
def load_ui_metadata_if_exists():
    if Path(UI_META_PATH).exists():
        return joblib.load(UI_META_PATH)
    return None


def build_features_row(user_input: dict, schema: dict) -> pd.DataFrame:
    cols = schema["feature_columns"]
    cat_cols = set(schema["cat_cols"])
    num_cols = set(schema.get("num_cols", []))

    row = {}
    for c in cols:
        if c in user_input:
            row[c] = user_input[c]
        else:
            row[c] = "Brak danych" if c in cat_cols else 0

    X_one = pd.DataFrame([row], columns=cols)

    # typy jak w treningu
    for c in cat_cols:
        if c in X_one.columns:
            X_one[c] = X_one[c].astype(str).fillna("Brak danych")

    for c in num_cols:
        if c in X_one.columns:
            X_one[c] = pd.to_numeric(X_one[c], errors="coerce").fillna(0)

    return X_one


def clean_choice(v: str) -> str:
    if v is None:
        return "Brak danych"
    v = str(v).strip()
    return "Brak danych" if v == "" or v == "[nie wybrano]" else v


def get_cat_options(ui_meta: dict | None, col: str, add_none: bool = True):
    if ui_meta and "cat_options" in ui_meta and col in ui_meta["cat_options"]:
        vals = ui_meta["cat_options"][col]
        return (["[nie wybrano]"] + vals) if add_none else vals
    return (["[nie wybrano]"] if add_none else [])


def get_num_stats(ui_meta: dict | None, col: str, fallback_min: int, fallback_max: int, fallback_med: int):
    if ui_meta and "num_stats" in ui_meta and col in ui_meta["num_stats"]:
        s = ui_meta["num_stats"][col]
        mn = int(s.get("min", fallback_min))
        mx = int(s.get("max", fallback_max))
        md = int(s.get("median", fallback_med))
        # sanity
        mx = max(mx, mn)
        md = min(max(md, mn), mx)
        return mn, mx, md
    return fallback_min, fallback_max, fallback_med


# ===== model / schema =====
try:
    model, schema = load_model_and_schema()
except FileNotFoundError:
    st.error(
        f"Brak modelu lub schematu:\n- {MODEL_PATH}\n- {SCHEMA_PATH}\n\n"
        "Najpierw uruchom `train_model.py`, żeby wytrenować i zapisać model."
    )
    st.stop()

ui_meta = load_ui_metadata_if_exists()

if ui_meta is None:
    st.info(
        "Nie znaleziono `models/ui_metadata.joblib`.\n\n"
        "Formularz będzie działał, ale część pól tekstowych trzeba wpisać ręcznie "
        "(bo bez CSV/meta nie wiemy jakie są dostępne kategorie)."
    )
else:
    st.caption("Załadowano `ui_metadata.joblib` – dropdowny będą sensowniejsze.")


# ===== UI =====
st.subheader("Formularz wyceny")

# Ustalamy zestaw “głównych” pól (reszta trafi do expandera)
main_fields = [
    "Vehicle_brand",
    "Vehicle_model",
    "Condition",
    "Production_year",
    "Mileage_km",
    "Fuel_type",
    "Power_HP",
    "Displacement_cm3",
    "Transmission",
    "Drive",
    "Type",
    "Doors_number",
    "Colour",
    "Origin_country",
    "First_owner",
    "Offer_location",
]

feature_cols = schema["feature_columns"]
cat_cols = set(schema["cat_cols"])
num_cols = set(schema.get("num_cols", []))

# Wspólna mapa inputów
user_input: dict = {}

# Marka / Model z mapą brand->models jeśli jest
c1, c2, c3 = st.columns(3)

with c1:
    if ui_meta and ui_meta.get("cat_options", {}).get("Vehicle_brand"):
        brand_options = ui_meta["cat_options"]["Vehicle_brand"]
        brand_form = st.selectbox("Marka (Vehicle_brand)", brand_options)
    else:
        brand_form = st.text_input("Marka (Vehicle_brand)", value="").strip() or "Brak danych"
    user_input["Vehicle_brand"] = brand_form

with c2:
    if ui_meta and ui_meta.get("brand_to_models") and brand_form in ui_meta["brand_to_models"]:
        model_options = ui_meta["brand_to_models"][brand_form]
        vehicle_model = st.selectbox("Model (Vehicle_model)", model_options) if model_options else "Brak danych"
    else:
        vehicle_model = st.text_input("Model (Vehicle_model)", value="").strip() or "Brak danych"
    user_input["Vehicle_model"] = vehicle_model

with c3:
    opts = get_cat_options(ui_meta, "Condition", add_none=False)
    if opts:
        condition = st.selectbox("Stan (Condition)", opts)
    else:
        condition = st.text_input("Stan (Condition)", value="").strip() or "Brak danych"
    user_input["Condition"] = condition

st.markdown("---")

with st.form("valuation_form"):
    left, right = st.columns(2)

    with left:
        mn, mx, md = get_num_stats(ui_meta, "Production_year", 1980, 2030, 2015)
        production_year = st.number_input("Rok produkcji (Production_year)", min_value=mn, max_value=mx, value=md, step=1)
        user_input["Production_year"] = int(production_year)

        mn, mx, md = get_num_stats(ui_meta, "Mileage_km", 0, 2_000_000, 150_000)
        mileage = st.number_input("Przebieg [km] (Mileage_km)", min_value=mn, max_value=mx, value=md, step=1000)
        user_input["Mileage_km"] = int(mileage)

        opts = get_cat_options(ui_meta, "Fuel_type", add_none=False)
        if opts:
            fuel_type = st.selectbox("Paliwo (Fuel_type)", opts)
        else:
            fuel_type = st.text_input("Paliwo (Fuel_type)", value="").strip() or "Brak danych"
        user_input["Fuel_type"] = fuel_type

    with right:
        mn, mx, md = get_num_stats(ui_meta, "Power_HP", 0, 1000, 120)
        power_hp = st.number_input("Moc [KM] (Power_HP)", min_value=mn, max_value=mx, value=md, step=10)
        user_input["Power_HP"] = float(power_hp)

        mn, mx, md = get_num_stats(ui_meta, "Displacement_cm3", 0, 8000, 1600)
        displacement = st.number_input("Pojemność [cm³] (Displacement_cm3)", min_value=mn, max_value=mx, value=md, step=100)
        user_input["Displacement_cm3"] = float(displacement)

        opts = get_cat_options(ui_meta, "Transmission", add_none=False)
        if opts:
            transmission = st.selectbox("Skrzynia (Transmission)", opts)
        else:
            transmission = st.text_input("Skrzynia (Transmission)", value="").strip() or "Brak danych"
        user_input["Transmission"] = transmission

    c4, c5, c6 = st.columns(3)
    with c4:
        opts = get_cat_options(ui_meta, "Drive", add_none=True)
        drive = st.selectbox("Napęd (Drive)", opts) if opts else st.text_input("Napęd (Drive)", value="")
        user_input["Drive"] = clean_choice(drive)

        opts = get_cat_options(ui_meta, "Type", add_none=True)
        car_type = st.selectbox("Nadwozie (Type)", opts) if opts else st.text_input("Nadwozie (Type)", value="")
        user_input["Type"] = clean_choice(car_type)

    with c5:
        mn, mx, md = get_num_stats(ui_meta, "Doors_number", 0, 10, 5)
        doors = st.number_input("Liczba drzwi (Doors_number)", min_value=mn, max_value=mx, value=md, step=1)
        user_input["Doors_number"] = int(doors)

        opts = get_cat_options(ui_meta, "Colour", add_none=True)
        colour = st.selectbox("Kolor (Colour)", opts) if opts else st.text_input("Kolor (Colour)", value="")
        user_input["Colour"] = clean_choice(colour)

    with c6:
        opts = get_cat_options(ui_meta, "Origin_country", add_none=True)
        origin = st.selectbox("Kraj pochodzenia (Origin_country)", opts) if opts else st.text_input("Kraj pochodzenia (Origin_country)", value="")
        user_input["Origin_country"] = clean_choice(origin)

        opts = get_cat_options(ui_meta, "First_owner", add_none=False)
        if opts:
            first_owner = st.selectbox("Pierwszy właściciel (First_owner)", opts)
        else:
            first_owner = st.text_input("Pierwszy właściciel (First_owner)", value="").strip() or "Brak danych"
        user_input["First_owner"] = clean_choice(first_owner)

    # Lokalizacja
    opts = get_cat_options(ui_meta, "Offer_location", add_none=False)
    if opts:
        offer_location = st.selectbox("Województwo (Offer_location)", opts)
    else:
        offer_location = st.text_input("Województwo (Offer_location)", value="").strip() or "Brak danych"
    user_input["Offer_location"] = offer_location

    # Pozostałe cechy z modelu (jeśli jakieś są) w expanderze
    remaining = [c for c in feature_cols if c not in user_input and c not in main_fields]
    if remaining:
        with st.expander("Pozostałe cechy (opcjonalne)"):
            st.caption("Jeśli model ma dodatkowe kolumny, możesz je tu wypełnić. Jak zostawisz puste, damy domyślne wartości.")
            for col in remaining:
                if col in cat_cols:
                    opts = get_cat_options(ui_meta, col, add_none=True)
                    if opts:
                        val = st.selectbox(f"{col}", opts, key=f"extra_{col}")
                    else:
                        val = st.text_input(f"{col}", value="", key=f"extra_{col}")
                    user_input[col] = clean_choice(val)
                else:
                    # num
                    mn, mx, md = get_num_stats(ui_meta, col, 0, 1_000_000, 0)
                    val = st.number_input(f"{col}", min_value=mn, max_value=mx, value=md, step=1, key=f"extra_{col}")
                    user_input[col] = float(val)

    submitted = st.form_submit_button("Wyceń")

if submitted:
    X_one = build_features_row(user_input, schema)
    pred = float(model.predict(X_one)[0])

    st.success("Wycena gotowa.")
    st.metric("Szacowana cena [PLN]", f"{pred:,.0f}".replace(",", " "))

    with st.expander("Pokaż dane wejściowe"):
        st.json(user_input)

    with st.expander("Pokaż wiersz cech przekazany do modelu"):
        st.dataframe(X_one)
