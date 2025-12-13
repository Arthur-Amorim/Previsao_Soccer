import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import os

# -----------------------------
# CONFIGURAÇÕES
# -----------------------------
BASE_URL = "https://www.cbf.com.br/futebol-brasileiro/calendario"
CSV_FILE = "jogos.csv"
LAST_RUN_FILE = "last_run.txt"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

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
# SCRAPER DO DIA
# -----------------------------
def scrape_day(date):
    url = f"{BASE_URL}?data={date.strftime('%Y-%m-%d')}"
    response = requests.get(url, headers=HEADERS, timeout=15)

    if response.status_code != 200:   # Verifica se a requisição foi bem-sucedida
        return []

    soup = BeautifulSoup(response.text, "lxml")
    jogos = []

    root = soup.select_one("div.styles_listGames__D8ydc")
    if not root:
        return jogos  # dia sem jogos

    for grupo in root.select("div.styles_groupGames__9XT_6"):
        # -------------------------
        # Campeonato
        # -------------------------
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
                # -------------------------
                # Times
                # -------------------------
                times = li.select("strong[title]")
                mandante = times[0]["title"].strip()
                visitante = times[1]["title"].strip()

                # -------------------------
                # Gols (se houver)
                # -------------------------
                gols = li.select("span.styles_gol__wQ4q9")
                gols_mandante = int(gols[0].text) if len(gols) > 0 else None
                gols_visitante = int(gols[1].text) if len(gols) > 1 else None

                # -------------------------
                # Infos adicionais
                # -------------------------
                info = li.select_one("div.styles_informations__K0gXK")

                numero_jogo = cidade = estado = estadio = hora = None

                if info:
                    num = info.select_one("span.styles_numberGame__dZ2sG")
                    numero_jogo = num.get_text(strip=True) if num else None

                    p = info.select_one("p")
                    if p:
                        linhas = [l.strip() for l in p.get_text("\n").split("\n")]

                        # 07/12/2025 - 16:00
                        if len(linhas) >= 1:
                            _, _, hora = linhas[0].partition(" - ")

                        # São Paulo - SP
                        if len(linhas) >= 2:
                            cidade, _, estado = linhas[1].partition(" - ")

                        # Estádio
                        if len(linhas) >= 3:
                            estadio = linhas[2]

                jogos.append({
                    "data": date.strftime("%Y-%m-%d"),
                    "campeonato": campeonato,
                    "numero_jogo": numero_jogo,
                    "mandante": mandante,
                    "visitante": visitante,
                    "gols_mandante": gols_mandante,
                    "gols_visitante": gols_visitante,
                    "hora": hora,
                    "cidade": cidade,
                    "estado": estado,
                    "estadio": estadio
                })

            except Exception as e:
                print("Erro ao processar jogo:", e)

    return jogos

# -----------------------------
# MAIN
# -----------------------------
def main():
    today = datetime.today().date()
    last_date = read_last_run_date()

    if last_date is None:
        print("Nenhuma data anterior encontrada. Defina last_run.txt.")
        return

    dates_to_run = list(daterange(last_date + timedelta(days=1), today))

    if not dates_to_run:
        print("Nenhuma data nova para processar.")
        return

    all_games = []

    for date in dates_to_run:
        print(f"Coletando jogos de {date}")
        jogos = scrape_day(date)
        all_games.extend(jogos)

    if not all_games:
        print("Nenhum jogo encontrado no período.")
        write_last_run_date(dates_to_run[-1])
        return

    df_new = pd.DataFrame(all_games)

    if os.path.exists(CSV_FILE):
        df_old = pd.read_csv(CSV_FILE)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_csv(CSV_FILE, index=False)

    write_last_run_date(dates_to_run[-1])

    print("Atualização concluída com sucesso.")


if __name__ == "__main__":
    main()


# Por algum motivo, ele não consegue entrar em for grupo in root.select("div.styles_groupGames__9XT_6"):
# para ler as informações dos jogos e campeonatos