import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import defaultdict

# --- 1. Carregar dados históricos ---
# Agora usando: datetime, home_goal, away_goal
df = pd.read_csv('Brasileirao_Matches.csv', parse_dates=['datetime'])

# Criar coluna de resultado (home win / draw / away win)
def match_result(row):
    if row['home_goal'] > row['away_goal']:
        return 'H'
    elif row['home_goal'] < row['away_goal']:
        return 'A'
    else:
        return 'D'

df['result'] = df.apply(match_result, axis=1)

# --- 2. Calcular ratings Elo simples ---
K = 20                 # fator de ajuste (ajustável)
initial_elo = 1500     # Elo inicial para todos os times
elos = defaultdict(lambda: initial_elo)
elo_history = []

def expected_score(home_elo, away_elo):
    return 1.0 / (1 + 10 ** ((away_elo - home_elo) / 400))

# Ordena por data e percorre jogo a jogo
for idx, row in df.sort_values('datetime').iterrows():
    home = row['home_team']
    away = row['away_team']

    home_elo = elos[home]
    away_elo = elos[away]

    exp_home = expected_score(home_elo, away_elo)
    exp_away = 1 - exp_home

    # resultado real convertido para pontuação Elo
    if row['result'] == 'H':
        score_home = 1
        score_away = 0
    elif row['result'] == 'A':
        score_home = 0
        score_away = 1
    else:  # empate
        score_home = 0.5
        score_away = 0.5

    # atualizar Elo
    elos[home] += K * (score_home - exp_home)
    elos[away] += K * (score_away - exp_away)

    elo_history.append({
        'datetime': row['datetime'],
        'season': row['season'],
        'round': row['round'],
        'home_team': home,
        'away_team': away,
        'home_elo_pre': home_elo,
        'away_elo_pre': away_elo,
        'result': row['result']
    })

elo_df = pd.DataFrame(elo_history)

# --- 3. Preparar dataset para ML (features + target) ---
elo_df['elo_diff'] = elo_df['home_elo_pre'] - elo_df['away_elo_pre']
elo_df['home_advantage'] = 1  # casa sempre conta como 1

X = elo_df[['elo_diff', 'home_advantage']]
y = elo_df['result']

# Encode resultado H/A/D → 0/1/2
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=1
)

# --- 4. Regressão Logística ---
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("Acurácia no teste:", model.score(X_test, y_test))

# --- 5. Função para prever novo jogo ---
def predict_match(home_team, away_team):
    home_elo = elos.get(home_team, initial_elo)
    away_elo = elos.get(away_team, initial_elo)

    elo_diff = home_elo - away_elo
    x = np.array([[elo_diff, 1]])

    probs = model.predict_proba(x)[0]

    # Índices: H = vitória casa, D = empate, A = vitória visitante
    p_home = probs[le.transform(['H'])[0]]
    p_draw = probs[le.transform(['D'])[0]]
    p_away = probs[le.transform(['A'])[0]]

    return {
        "Chance de vitória do time da casa": float(p_home),
        "Chance de empate": float(p_draw),
        "Chance de vitória do time visitante": float(p_away)
    }

# Exemplo:
print(predict_match('Fluminense-RJ','Vasco da Gama-RJ'))
