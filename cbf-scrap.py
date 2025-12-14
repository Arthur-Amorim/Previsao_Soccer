import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import os
import time
import unicodedata

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# -----------------------------
# CONFIGURAÇÕES
# -----------------------------
BASE_URL = "https://www.cbf.com.br/futebol-brasileiro/calendario"
CSV_FILE = "jogos.csv"
LAST_RUN_FILE = "last_run.txt"

MAPEAMENTO_TIMES = {
    # MG
    "Atlético Mineiro Saf": "Atlético-MG",
    "Cruzeiro Saf": "Cruzeiro",
    "America Saf": "América",

    # BA
    "Bahia": "Bahia",
    "Vitória": "Vitória",

    # RJ
    "Botafogo": "Botafogo",
    "Flamengo": "Flamengo",
    "Fluminense": "Fluminense",
    "Vasco da Gama S.a.f.": "Vasco da Gama",

    # SP
    "Corinthians": "Corinthians",
    "Palmeiras": "Palmeiras",
    "Santos Fc": "Santos",
    "São Paulo": "São Paulo",
    "Mirassol": "Mirassol",
    "Red Bull Bragantino": "Red Bull Bragantino",

    # RS
    "Grêmio": "Grêmio",
    "Internacional": "Internacional",
    "Juventude": "Juventude",

    # CE
    "Ceará": "Ceará",
    "Fortaleza": "Fortaleza",
    "Fortaleza Ec Saf": "Fortaleza",

    # PE
    "Sport Recife": "Sport",
}



# -----------------------------
# SELENIUM SETUP
# -----------------------------
def get_driver():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # ative em produção
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# -----------------------------
# FUNÇÕES AUXILIARES
# -----------------------------
def read_last_run_date():
    if not os.path.exists(LAST_RUN_FILE):
        return None
    with open(LAST_RUN_FILE, "r") as f:
        return datetime.strptime(f.read().strip(), "%Y-%m-%d").date()

def write_last_run_date(date):
    with open(LAST_RUN_FILE, "w") as f:
        f.write(date.strftime("%Y-%m-%d"))

def daterange(start_date, end_date):
    for n in range((end_date - start_date).days):
        yield start_date + timedelta(days=n)

# -----------------------------
# SCRAPER DO DIA (SELENIUM)
# -----------------------------
def scrape_day(driver, date):
    url = f"{BASE_URL}?data={date.strftime('%Y-%m-%d')}"
    driver.get(url)

    try:
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div[class*='styles_listGames']")
            )
        )
    except:
        return []

    time.sleep(1)

    soup = BeautifulSoup(driver.page_source, "lxml")
    jogos = []

    root = soup.select_one("div[class*='styles_listGames']")
    if not root:
        return jogos

    grupos = root.select("div[class*='styles_groupGames']")

    for grupo in grupos:
        h2 = grupo.select_one("h2")
        campeonato = h2.get_text(strip=True) if h2 else None

        splide = grupo.select_one("div.splide")
        if not splide:
            continue

        ul = splide.select_one("ul.splide__list")
        if not ul:
            continue

        for li in ul.select("li.splide__slide"):
            try:
                times = li.select("strong[title]")
                mandante = times[0]["title"].strip()
                visitante = times[1]["title"].strip()

                gols = li.select("span[class*='styles_gol']")
                gols_mandante = gols[0].text if len(gols) > 0 else None
                gols_visitante = gols[1].text if len(gols) > 1 else None

                info = li.select_one("div[class*='styles_informations']")
                hora = None

                if info:
                    p = info.select_one("p")
                    if p:
                        linhas = [l.strip() for l in p.get_text("\n").split("\n")]
                        if linhas:
                            _, _, hora = linhas[0].partition(" - ")

                jogos.append({
                    "data": date.strftime("%Y-%m-%d"),
                    "campeonato": campeonato,
                    "mandante": mandante,
                    "visitante": visitante,
                    "gols_mandante": gols_mandante,
                    "gols_visitante": gols_visitante,
                    "hora": hora
                })

            except Exception as e:
                print("Erro ao processar jogo:", e)

    return jogos

def padroniza_time(nome):
    if not isinstance(nome, str):
        return None

    nome = nome.strip()

    nome_padrao = MAPEAMENTO_TIMES.get(nome)

    if nome_padrao is None:
        print(f"[AVISO] Time não mapeado: '{nome}'")
        return remove_acentos(nome)  # também remove acento do original

    return remove_acentos(nome_padrao)

def remove_acentos(texto):
    if not isinstance(texto, str):
        return texto
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


# -----------------------------
# MAIN
# -----------------------------
def main():
    today = (datetime.today() + timedelta(days=1)).date()
    last_date = read_last_run_date()

    if last_date is None:
        print("Nenhuma data anterior encontrada.")
        return

    dates_to_run = list(daterange(last_date + timedelta(days=1), today))

    if not dates_to_run:
        print("Nenhuma data nova para processar.")
        return

    driver = get_driver()
    all_games = []

    try:
        for date in dates_to_run:
            print(f"Coletando jogos de {date}")
            jogos = scrape_day(driver, date)
            all_games.extend(jogos)
    finally:
        driver.quit()

    if not all_games:
        print("Nenhum jogo encontrado.")
        write_last_run_date(dates_to_run[-1])
        return

    df_new = pd.DataFrame(all_games)

    # -----------------------------
    # FORMATO FINAL SOLICITADO
    # -----------------------------
    df_new["gols_mandante"] = pd.to_numeric(df_new["gols_mandante"], errors="coerce")
    df_new["gols_visitante"] = pd.to_numeric(df_new["gols_visitante"], errors="coerce")

    df_new["Country"] = "Brazil"

    def normaliza_liga(nome):
        if not isinstance(nome, str):
            return None
        nome = nome.lower()
        if "série a" in nome or "serie a" in nome:
            return "Serie A"
        if "série b" in nome or "serie b" in nome:
            return "Serie B"
        return nome.title()

    df_new["League"] = df_new["campeonato"].apply(normaliza_liga)
    df_new["Year"] = pd.to_datetime(df_new["data"]).dt.year.astype(float)

    df_new["Date"] = pd.to_datetime(df_new["data"]).apply(
        lambda d: f"{d.day}/{d.month}/{d.year}"
    )

    df_new["Time"] = df_new["hora"].fillna("")
    df_new["Home"] = df_new["mandante"].apply(padroniza_time)
    df_new["Away"] = df_new["visitante"].apply(padroniza_time)
    df_new["HG"] = df_new["gols_mandante"].astype(float)
    df_new["AG"] = df_new["gols_visitante"].astype(float)

    def resultado(row):
        if pd.isna(row["HG"]) or pd.isna(row["AG"]):
            return None
        if row["HG"] > row["AG"]:
            return "H"
        elif row["HG"] < row["AG"]:
            return "A"
        else:
            return "D"

    df_new["Res"] = df_new.apply(resultado, axis=1)

    df_new = df_new[
        ["Country", "League", "Year", "Date", "Time", "Home", "Away", "HG", "AG", "Res"]
    ]

    # -----------------------------
    # APPEND SEGURO AO CSV (RECOMENDADO)
    # -----------------------------
    # -----------------------------
    # APPEND REAL AO CSV (SEM SOBRESCREVER)
    # -----------------------------
    file_exists = os.path.exists(CSV_FILE)

    df_new.to_csv(
        CSV_FILE,
        mode="a",                 # append
        header=not file_exists,   # escreve cabeçalho só se o arquivo NÃO existir
        index=False,
        encoding="utf-8-sig"
    )

    write_last_run_date(dates_to_run[-1])

    print("Atualização concluída com sucesso.")
    print(f"{len(df_new)} linhas adicionadas ao CSV.")
    write_last_run_date(dates_to_run[-1])


main()


####################################################
# Tarefa: Buscar Odds em sites de aposta para comparar com o modelo preditivo
    # https://www.oddsagora.com.br/football/brazil/copa-betano-do-brasil/
    # https://www.betbrain.com/football-betting-odds/brazil
# Anotações: para funcionar deve conter arquivo jogos.csv minimamente preenchido e last_run com data da ultima atualização
# Data: 14/12/2025
# Autor: Arthur Amorim 
####################################################

