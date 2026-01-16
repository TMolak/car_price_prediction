import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Wycena samochodu", layout="wide")
st.title("Wycena samochodu (ML) + przegląd ogłoszeń")

st.write(
    """
Masz dane auta, klikasz, dostajesz wycenę.  
To estymacja na podstawie danych ogłoszeniowych, a nie przepowiednia rynku.
"""
)

DATA_PATH = "data/Car_sale_ads_cleaned_v2.csv"
MODEL_PATH = "models/catboost_price.joblib"
SCHEMA_PATH = "models/feature_schema.joblib"


@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # przyjemne rzutowanie (opcjonalne)
    for col in ["Offer_location", "Vehicle_brand", "Condition", "Fuel_type"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


@st.cache_resource
def load_model_and_schema():
    model = joblib.load(MODEL_PATH)
    schema = joblib.load(SCHEMA_PATH)
    return model, schema


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


# ===== wczytanie danych =====
try:
    df = load_data()
except FileNotFoundError:
    st.error(
        f"Nie znaleziono pliku `{DATA_PATH}`. Najpierw uruchom `clean_data.py`."
    )
    st.stop()

# ===== model =====
model_ok = True
try:
    model, schema = load_model_and_schema()
except FileNotFoundError:
    model_ok = False
    st.warning(
        f"Nie znaleziono modelu `{MODEL_PATH}` lub schema `{SCHEMA_PATH}`.\n\n"
        "Uruchom `train_model.py`, żeby wytrenować i zapisać model."
    )


# ===== sidebar filtry =====
st.sidebar.header("Filtry danych (tabela)")

filtered = df.copy()

# Marka
selected_brand = "[wszystkie]"
if "Vehicle_brand" in df.columns:
    brands = ["[wszystkie]"] + sorted(df["Vehicle_brand"].dropna().unique().tolist())
    selected_brand = st.sidebar.selectbox("Marka:", brands)
    if selected_brand != "[wszystkie]":
        filtered = filtered[filtered["Vehicle_brand"] == selected_brand]

# Model
selected_model = "[wszystkie]"
if "Vehicle_model" in df.columns:
    model_source = filtered if selected_brand != "[wszystkie]" else df
    models = ["[wszystkie]"] + sorted(model_source["Vehicle_model"].dropna().unique().tolist())
    selected_model = st.sidebar.selectbox("Model:", models)
    if selected_model != "[wszystkie]":
        filtered = filtered[filtered["Vehicle_model"] == selected_model]

# Stan
if "Condition" in df.columns:
    conditions = ["[wszystkie]"] + sorted(df["Condition"].dropna().unique().tolist())
    selected_condition = st.sidebar.selectbox("Stan pojazdu:", conditions)
    if selected_condition != "[wszystkie]":
        filtered = filtered[filtered["Condition"] == selected_condition]

# Paliwo
if "Fuel_type" in df.columns:
    fuels = ["[wszystkie]"] + sorted(df["Fuel_type"].dropna().unique().tolist())
    selected_fuel = st.sidebar.selectbox("Rodzaj paliwa:", fuels)
    if selected_fuel != "[wszystkie]":
        filtered = filtered[filtered["Fuel_type"] == selected_fuel]

# Województwo
if "Offer_location" in df.columns:
    locs = ["[wszystkie]"] + sorted(df["Offer_location"].dropna().unique().tolist())
    selected_loc = st.sidebar.selectbox("Województwo:", locs)
    if selected_loc != "[wszystkie]":
        filtered = filtered[filtered["Offer_location"] == selected_loc]

# Zakres ceny
if "Price" in df.columns and filtered["Price"].notna().any():
    min_price = int(df["Price"].min())
    max_price = int(df["Price"].max())
    price_min, price_max = st.sidebar.slider(
        "Cena [PLN]",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=1000,
    )
    filtered = filtered[(filtered["Price"] >= price_min) & (filtered["Price"] <= price_max)]

# Rok produkcji
if "Production_year" in df.columns and df["Production_year"].notna().any():
    min_year = int(df["Production_year"].min())
    max_year = int(df["Production_year"].max())
    year_min, year_max = st.sidebar.slider(
        "Rok produkcji:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
    )
    filtered = filtered[(filtered["Production_year"] >= year_min) & (filtered["Production_year"] <= year_max)]

# Przebieg
if "Mileage_km" in df.columns and df["Mileage_km"].notna().any():
    min_mileage = int(df["Mileage_km"].min())
    max_mileage = int(df["Mileage_km"].max())
    mileage_min, mileage_max = st.sidebar.slider(
        "Przebieg [km]:",
        min_value=min_mileage,
        max_value=max_mileage,
        value=(min_mileage, max_mileage),
        step=1000,
    )
    filtered = filtered[(filtered["Mileage_km"] >= mileage_min) & (filtered["Mileage_km"] <= mileage_max)]

st.sidebar.markdown(f"**Liczba ogłoszeń po filtrach:** {len(filtered)}")


# ===== tabs =====
tab_data, tab_form = st.tabs(["Dane (tabela)", "Wycena auta (formularz)"])

with tab_data:
    st.subheader("Tabela ogłoszeń (po filtrach)")
    all_cols = list(filtered.columns)
    default_cols = [c for c in ["Price", "Vehicle_brand", "Vehicle_model",
                                "Production_year", "Mileage_km", "Fuel_type",
                                "Condition", "Offer_location"] if c in all_cols]
    selected_cols = st.multiselect(
        "Wybierz kolumny do wyświetlenia:",
        options=all_cols,
        default=default_cols if default_cols else all_cols[:10]
    )
    st.dataframe(filtered[selected_cols] if selected_cols else filtered)


with tab_form:
    st.subheader("Formularz wyceny samochodu")

    if not model_ok:
        st.info("Najpierw wytrenuj model (`train_model.py`), potem wróć tutaj.")
        st.stop()

    # Opcje z danych, żeby nie strzelać wartościami typu "tak/nie" jeśli w danych jest "Yes/No"
    def options(col: str, add_none: bool = True):
        if col in df.columns:
            vals = sorted(df[col].dropna().astype(str).unique().tolist())
            return (["[nie wybrano]"] + vals) if add_none else vals
        return ["[nie wybrano]"] if add_none else []

    st.markdown("### Podstawowe dane")

    c1, c2, c3 = st.columns(3)
    with c1:
        brand_form = st.selectbox("Marka (Vehicle_brand):", options("Vehicle_brand", add_none=False))
    with c2:
        if "Vehicle_model" in df.columns and brand_form:
            model_source = df[df["Vehicle_brand"].astype(str) == str(brand_form)]
            model_opts = sorted(model_source["Vehicle_model"].dropna().astype(str).unique().tolist())
        else:
            model_opts = []
        model_form = st.selectbox("Model (Vehicle_model):", model_opts if model_opts else ["Brak danych"])
    with c3:
        car_type = st.selectbox("Typ nadwozia (Type):", options("Type"))

    st.markdown("---")

    with st.form("car_valuation_form"):
        left, right = st.columns(2)

        with left:
            year = st.number_input(
                "Rok produkcji (Production_year):",
                min_value=int(df["Production_year"].min()) if "Production_year" in df.columns else 1980,
                max_value=int(df["Production_year"].max()) if "Production_year" in df.columns else 2030,
                value=int(df["Production_year"].median()) if "Production_year" in df.columns else 2015,
                step=1,
            )
            mileage = st.number_input(
                "Przebieg [km] (Mileage_km):",
                min_value=0,
                max_value=int(df["Mileage_km"].max()) if "Mileage_km" in df.columns else 2_000_000,
                value=int(df["Mileage_km"].median()) if "Mileage_km" in df.columns else 150_000,
                step=1000,
            )

        with right:
            power = st.number_input(
                "Moc [KM] (Power_HP):",
                min_value=0,
                max_value=int(df["Power_HP"].max()) if "Power_HP" in df.columns and df["Power_HP"].notna().any() else 1000,
                value=int(df["Power_HP"].median()) if "Power_HP" in df.columns and df["Power_HP"].notna().any() else 120,
                step=10,
            )
            displacement = st.number_input(
                "Pojemność [cm³] (Displacement_cm3):",
                min_value=0,
                max_value=int(df["Displacement_cm3"].max()) if "Displacement_cm3" in df.columns and df["Displacement_cm3"].notna().any() else 8000,
                value=int(df["Displacement_cm3"].median()) if "Displacement_cm3" in df.columns and df["Displacement_cm3"].notna().any() else 1600,
                step=100,
            )

        c4, c5, c6 = st.columns(3)

        with c4:
            fuel = st.selectbox("Rodzaj paliwa (Fuel_type):", options("Fuel_type", add_none=False))
            drive = st.selectbox("Napęd (Drive):", options("Drive"))

        with c5:
            transmission = st.selectbox("Skrzynia biegów (Transmission):", options("Transmission", add_none=False))
            doors_number = st.number_input("Liczba drzwi (Doors_number):", min_value=0, max_value=10, value=5, step=1)

        with c6:
            colour = st.selectbox("Kolor (Colour):", options("Colour"))
            condition = st.selectbox("Stan (Condition):", options("Condition", add_none=False))

        c7, c8, c9 = st.columns(3)

        with c7:
            origin_country = st.selectbox("Kraj pochodzenia (Origin_country):", options("Origin_country"))
            first_owner = st.selectbox("Pierwszy właściciel (First_owner):", options("First_owner", add_none=False) or ["Brak danych"])

        with c8:
            location = st.selectbox("Lokalizacja (Offer_location):", options("Offer_location", add_none=False))

        with c9:
            st.caption("Cena nie jest potrzebna do wyceny. To target.")
            dummy = st.text_input("Uwagi (opcjonalnie, nie wpływa na model):", value="")

        submitted = st.form_submit_button("Wyceń")

        if submitted:
            # wartości "[nie wybrano]" -> "Brak danych"
            def clean_choice(v):
                return "Brak danych" if v in (None, "", "[nie wybrano]") else v

            user_input = {
                "Condition": condition,
                "Vehicle_brand": brand_form,
                "Vehicle_model": model_form,
                "Production_year": int(year),
                "Mileage_km": int(mileage),
                "Power_HP": float(power),
                "Displacement_cm3": float(displacement),
                "Fuel_type": fuel,
                "Drive": clean_choice(drive),
                "Transmission": transmission,
                "Type": clean_choice(car_type),
                "Doors_number": int(doors_number),
                "Colour": clean_choice(colour),
                "Origin_country": clean_choice(origin_country),
                "First_owner": clean_choice(first_owner),
                "Offer_location": location,
            }

            X_one = build_features_row(user_input, schema)
            pred = float(model.predict(X_one)[0])

            st.success("Wycena gotowa.")
            st.metric("Szacowana cena [PLN]", f"{pred:,.0f}".replace(",", " "))

            with st.expander("Pokaż dane wejściowe"):
                st.json(user_input)

            with st.expander("Pokaż wiersz cech przekazany do modelu"):
                st.dataframe(X_one)
