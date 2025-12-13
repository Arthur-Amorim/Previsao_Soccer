import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# =========================
# 1. Carregar dados
# =========================
df = pd.read_excel("BRA.xlsx")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# =========================
# 2. Criar target binário
# =========================
df['home_win'] = (df['HG'] > df['AG']).astype(int)

# =========================
# 3. Inicializar Elo
# =========================
teams = pd.concat([df['Home'], df['Away']]).unique()
elo = {team: 1500 for team in teams}
K = 20

def expected_score(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

elo_diff_list = []

# =========================
# 4. Loop temporal (SEM leakage)
# =========================
for _, row in df.iterrows():
    home, away = row['Home'], row['Away']

    # Feature ANTES do jogo
    elo_diff_list.append(elo[home] - elo[away])

    # Resultado
    if row['HG'] > row['AG']:
        s_home, s_away = 1, 0
    elif row['HG'] < row['AG']:
        s_home, s_away = 0, 1
    else:
        s_home, s_away = 0.5, 0.5

    e_home = expected_score(elo[home], elo[away])
    e_away = expected_score(elo[away], elo[home])

    # Atualiza Elo APÓS o jogo
    elo[home] += K * (s_home - e_home)
    elo[away] += K * (s_away - e_away)

df['elo_diff'] = elo_diff_list

X = df[['elo_diff']]
y = df['home_win']

# =========================
# 5. Split temporal
# =========================
split = int(0.8 * len(df))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# =========================
# 6. Modelos
# =========================
models = {
    'LR': LogisticRegression(),
    'RF': RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', probability=True))
    ]),
    'XGB': XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )
}

# Treinar modelos
for model in models.values():
    model.fit(X_train, y_train)

# =========================
# 7. Função de predição
# =========================
def predict_match(home_team, away_team):
    """
    Retorna a probabilidade de vitória do time da casa
    """
    if home_team not in elo or away_team not in elo:
        raise ValueError("Time não encontrado na base de dados.")

    elo_diff = elo[home_team] - elo[away_team]
    X_pred = pd.DataFrame({'elo_diff': [elo_diff]})

    results = {}
    for name, model in models.items():
        prob = model.predict_proba(X_pred)[0, 1]
        results[name] = round(prob, 3)

    return results

print(predict_match("Corinthians", "Cruzeiro"))