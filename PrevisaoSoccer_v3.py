import pandas as pd
import numpy as np
from collections import defaultdict, deque

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

# =========================================================
# 1. CARREGAR DADOS
# =========================================================
df = pd.read_excel("BRA.xlsx")

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

# Target binário: vitória do mandante
df['home_win'] = (df['HG'] > df['AG']).astype(int)

# =========================================================
# 2. INICIALIZAÇÕES
# =========================================================
teams = pd.concat([df['Home'], df['Away']]).unique()

# Elo
elo = {team: 1500.0 for team in teams}
K = 20

# Histórico últimos 5 jogos
goal_diff_hist = defaultdict(lambda: deque(maxlen=5))
result_hist = defaultdict(lambda: deque(maxlen=5))
date_hist = defaultdict(lambda: deque(maxlen=5))

# =========================================================
# 3. FUNÇÕES AUXILIARES (ROBUSTAS)
# =========================================================
def expected_score(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

def mean_or_default(values, default=0.0):
    return np.mean(values) if len(values) > 0 else default

def mean_rest_days(dates):
    if len(dates) < 2:
        return 7.0

    diffs = []
    for i in range(1, len(dates)):
        if pd.isna(dates[i]) or pd.isna(dates[i - 1]):
            continue
        delta = (dates[i] - dates[i - 1]).days
        if delta is not None and not np.isnan(delta):
            diffs.append(delta)

    return np.mean(diffs) if len(diffs) > 0 else 7.0

# =========================================================
# 4. FEATURE ENGINEERING TEMPORAL
# =========================================================
features = []

for _, row in df.iterrows():
    home, away = row['Home'], row['Away']
    match_date = row['Date']

    # -------- FEATURES ANTES DO JOGO --------
    features.append({
        'elo_diff': elo[home] - elo[away],

        'home_gd_5': mean_or_default(goal_diff_hist[home]),
        'away_gd_5': mean_or_default(goal_diff_hist[away]),

        'home_form_5': mean_or_default(result_hist[home], 0.5),
        'away_form_5': mean_or_default(result_hist[away], 0.5),

        'home_rest_5': mean_rest_days(list(date_hist[home])),
        'away_rest_5': mean_rest_days(list(date_hist[away]))
    })

    # -------- RESULTADO DO JOGO --------
    if row['HG'] > row['AG']:
        s_home, s_away = 1, 0
        r_home, r_away = 1, 0
    elif row['HG'] < row['AG']:
        s_home, s_away = 0, 1
        r_home, r_away = 0, 1
    else:
        s_home, s_away = 0.5, 0.5
        r_home, r_away = 0.5, 0.5

    # -------- ATUALIZAR HISTÓRICOS --------
    goal_diff_hist[home].append(row['HG'] - row['AG'])
    goal_diff_hist[away].append(row['AG'] - row['HG'])

    result_hist[home].append(r_home)
    result_hist[away].append(r_away)

    date_hist[home].append(match_date)
    date_hist[away].append(match_date)

    # -------- ATUALIZAR ELO --------
    e_home = expected_score(elo[home], elo[away])
    e_away = expected_score(elo[away], elo[home])

    elo[home] += K * (s_home - e_home)
    elo[away] += K * (s_away - e_away)

# =========================================================
# 5. DATASET FINAL
# =========================================================
X = pd.DataFrame(features)
y = df['home_win']

# =========================================================
# 6. SPLIT TEMPORAL
# =========================================================
split = int(0.8 * len(df))

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# =========================================================
# 7. MODELOS (COM IMPUTER)
# =========================================================
models = {
    'LR': Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('lr', LogisticRegression(max_iter=500))
    ]),

    'RF': RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42
    ),

    'SVM': Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
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

for model in models.values():
    model.fit(X_train, y_train)

# =========================================================
# 8. FUNÇÃO DE PREVISÃO
# =========================================================
def predict_match(home_team, away_team):
    """
    Retorna a probabilidade de vitória do time da casa
    segundo cada modelo.
    """
    if home_team not in elo or away_team not in elo:
        raise ValueError("Time não encontrado na base de dados.")

    input_features = pd.DataFrame([{
        'elo_diff': elo[home_team] - elo[away_team],

        'home_gd_5': mean_or_default(goal_diff_hist[home_team]),
        'away_gd_5': mean_or_default(goal_diff_hist[away_team]),

        'home_form_5': mean_or_default(result_hist[home_team], 0.5),
        'away_form_5': mean_or_default(result_hist[away_team], 0.5),

        'home_rest_5': mean_rest_days(list(date_hist[home_team])),
        'away_rest_5': mean_rest_days(list(date_hist[away_team]))
    }])

    return {
        name: round(model.predict_proba(input_features)[0, 1], 3)
        for name, model in models.items()
    }

# =========================================================
# 9. EXEMPLO DE USO
# =========================================================
print(predict_match("Corinthians", "Cruzeiro"))


# Tarefas: Implementar mais parametros 
# Tarefas: Arrumar mais dados 
# Tarefas: Escrever código que atualiza dados automáticamente