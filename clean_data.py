import re
import pandas as pd


VOIVODESHIPS = {
    "dolnośląskie":        ["dolnośląskie", "dolnoslaskie", "dolno slaskie", "dolno-slaskie", "dolnoślaskie"],
    "kujawsko-pomorskie":  ["kujawsko-pomorskie", "kujawsko pomorskie", "kuj pom", "kujawsko-pom"],
    "lubelskie":           ["lubelskie"],
    "lubuskie":            ["lubuskie"],
    "łódzkie":             ["łódzkie", "lodzkie", "łodzkie"],
    "małopolskie":         ["małopolskie", "malopolskie", "małopolska", "malopolska", "mało polskie"],
    "mazowieckie":         ["mazowieckie", "mazowsze"],
    "opolskie":            ["opolskie"],
    "podkarpackie":        ["podkarpackie"],
    "podlaskie":           ["podlaskie"],
    "pomorskie":           ["pomorskie", "pomorze"],
    "śląskie":             ["śląskie", "slaskie", "ślaskie"],
    "świętokrzyskie":      ["świętokrzyskie", "swietokrzyskie", "św tok", "św-krzyskie"],
    "warmińsko-mazurskie": ["warmińsko-mazurskie", "warminsko-mazurskie", "warminsko mazurskie", "warmińsko mazurskie"],
    "wielkopolskie":       ["wielkopolskie", "wielkopolska"],
    "zachodniopomorskie":  ["zachodniopomorskie", "zachodnio pomorskie", "zachodnio-pomorskie"],
}


def _normalize_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[()\-_/]", " ", t)
    t = re.sub(r"\b(polska|poland)\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_voivodeship(text) -> str:
    if pd.isna(text):
        return "Brak danych"
    t = _normalize_text(str(text))
    for proper_name, variants in VOIVODESHIPS.items():
        for variant in variants:
            if variant in t:
                return proper_name
    return "Brak danych"


def main(
    input_path: str = "data/Car_sale_ads.csv",
    output_path: str = "data/Car_sale_ads_cleaned_v2.csv",
    eur_rate: float = 4.6,
):
    df_raw = pd.read_csv(input_path)
    df = df_raw.copy()

    # Lokalizacja -> województwo
    if "Offer_location" in df.columns:
        df["Offer_location"] = df["Offer_location"].apply(extract_voivodeship).astype("category")

    # Usuwanie wybranych kolumn jeśli istnieją
    cols_to_drop = ["Vehicle_version", "CO2_emissions", "First_registration_date", "Vehicle_generation"]
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if existing_cols_to_drop:
        df.drop(columns=existing_cols_to_drop, inplace=True, errors="ignore")
        print(f"[OK] Usunięto kolumny: {existing_cols_to_drop}")
    else:
        print("[INFO] Brak kolumn do usunięcia z listy.")

    # Uzupełnianie braków w kluczowych kategorycznych
    for col in ["Origin_country", "First_owner", "Drive"]:
        if col in df.columns:
            df[col] = df[col].fillna("Brak danych")
        else:
            print(f"[INFO] Brak kolumny '{col}' w danych.")

    # Doors_number -> int
    if "Doors_number" in df.columns:
        df["Doors_number"] = (
            df["Doors_number"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
        )
        df["Doors_number"] = df["Doors_number"].fillna(0).astype(int)

    # Waluta EUR -> PLN
    if "Currency" in df.columns and "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce").astype(float)

        eur_mask = df["Currency"].astype(str).str.upper() == "EUR"
        n_eur = int(eur_mask.sum())
        print(f"[INFO] Liczba ogłoszeń w EUR przed konwersją: {n_eur}")

        df.loc[eur_mask, "Price"] = df.loc[eur_mask, "Price"] * float(eur_rate)
        df.loc[eur_mask, "Currency"] = "PLN"

        print(f"[OK] Zamieniono EUR -> PLN po kursie {eur_rate}")
    else:
        print("[INFO] Brak Currency/Price, pomijam konwersję waluty.")

    df.to_csv(output_path, index=False)
    print(f"[OK] Zapisano: {output_path} | shape={df.shape}")


if __name__ == "__main__":
    main()
