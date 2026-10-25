# Vehicle Registration Trend Analysis
# Run:  python vehicle_analysis.py
#
# What this program does:
#   1. loads the registration data
#   2. finds trends (yearly growth, fuel types, states, busy months)
#   3. ML: forecasts the next 12 months (Linear Regression vs Random Forest)
#   4. ML: groups the states with KMeans
# Charts are saved in the "charts" folder.
#
# NOTE: the data in data/registrations_synthetic.csv is MADE UP (made with make_sample_data.py)
# so the results only show how the method works. Real data in the same columns will also work.

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # so charts can be saved without opening a window
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

os.makedirs("charts", exist_ok=True)


def save_chart(name):
    plt.tight_layout()
    plt.savefig("charts/" + name + ".png", dpi=120)
    plt.close()
    print("saved charts/" + name + ".png")


# ---------------------------------------------------------------
# Step 1: load the data
# ---------------------------------------------------------------
df = pd.read_csv("data/registrations_synthetic.csv")
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

print(df.head())
print("rows and columns:", df.shape)
print("missing values:", df.isnull().sum().sum())
print("states:", df["state"].nunique(), "| vehicle types:", df["vehicle_class"].nunique(), "| fuels:", df["fuel"].nunique())
print("data from", df["date"].min().date(), "to", df["date"].max().date())

# total registrations of all vehicles for every month
monthly = df.groupby("date")["registrations"].sum()
last_date = monthly.index.max()

# ---------------------------------------------------------------
# Step 2: trends
# ---------------------------------------------------------------
# Chart 1: total registrations every month
plt.figure(figsize=(10, 5))
plt.plot(monthly.index, monthly.values / 1e6, label="monthly total")
plt.plot(monthly.index, monthly.rolling(12).mean() / 1e6, color="red", label="12 month average")
plt.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-08-01"), color="grey", alpha=0.2)
plt.text(pd.Timestamp("2020-04-01"), 1.4, "lockdown period")
plt.title("Total vehicle registrations per month")
plt.xlabel("Month")
plt.ylabel("Registrations (millions)")
plt.legend()
plt.grid(True)
save_chart("1_monthly_trend")

# Chart 2: yearly registrations by vehicle type (2026 is skipped because it only has 9 months)
by_year = df[df["year"] <= 2025].pivot_table(index="year", columns="vehicle_class", values="registrations", aggfunc="sum")
by_year.plot(kind="bar", stacked=True, figsize=(10, 5))
plt.title("Yearly registrations by vehicle type")
plt.xlabel("Year")
plt.ylabel("Registrations")
plt.xticks(rotation=0)
plt.grid(axis="y")
save_chart("2_yearly_by_type")

yearly_total = by_year.sum(axis=1)
yearly_growth = yearly_total.pct_change() * 100
print("\nYearly total (millions):")
print((yearly_total / 1e6).round(2))
print("Yearly growth (%):")
print(yearly_growth.round(1))

# Chart 3: share of each fuel type by year
fuel_year = df.pivot_table(index="year", columns="fuel", values="registrations", aggfunc="sum")
fuel_share = fuel_year.div(fuel_year.sum(axis=1), axis=0) * 100
fuel_share.plot(marker="o", figsize=(10, 5))
plt.title("Share of each fuel type (%)")
plt.xlabel("Year")
plt.ylabel("Share of new registrations (%)")
plt.grid(True)
save_chart("3_fuel_share")
print("\nFuel share by year (%):")
print(fuel_share.round(1))

# Chart 4: electric share for each vehicle type
ev = df[df["fuel"] == "Electric"].pivot_table(index="year", columns="vehicle_class", values="registrations", aggfunc="sum")
all_v = df.pivot_table(index="year", columns="vehicle_class", values="registrations", aggfunc="sum")
ev_share = ev / all_v * 100
ev_share.plot(marker="o", figsize=(10, 5))
plt.title("Electric share by vehicle type (%)")
plt.xlabel("Year")
plt.ylabel("Electric share (%)")
plt.grid(True)
save_chart("4_ev_share_by_type")
print("\nEV share by vehicle type (%):")
print(ev_share.round(1))

# Chart 5: registrations by state in the last 12 months
recent = df[df["date"] > last_date - pd.DateOffset(months=12)]
before = df[(df["date"] <= last_date - pd.DateOffset(months=12)) & (df["date"] > last_date - pd.DateOffset(months=24))]
state_recent = recent.groupby("state")["registrations"].sum().sort_values()
plt.figure(figsize=(10, 5))
plt.barh(state_recent.index, state_recent.values / 1e6)
plt.title("Registrations by state (last 12 months)")
plt.xlabel("Registrations (millions)")
plt.grid(axis="x")
save_chart("5_states")
print("\nTop states (millions):")
print((state_recent.sort_values(ascending=False) / 1e6).round(2))

# Chart 6: which months are busy? (only normal years 2022 to 2025)
normal = monthly[(monthly.index.year >= 2022) & (monthly.index.year <= 2025)]
ratio = normal / normal.groupby(normal.index.year).transform("mean")   # month compared with its year average
month_pattern = (ratio.groupby(ratio.index.month).mean() - 1) * 100
plt.figure(figsize=(10, 5))
plt.bar(month_pattern.index, month_pattern.values)
plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Busy and slow months (compared with the average month)")
plt.ylabel("% above / below average")
plt.grid(axis="y")
save_chart("6_month_pattern")
print("\nMonth pattern (% vs average month):")
print(month_pattern.round(1))

# ---------------------------------------------------------------
# Step 3: ML - forecast the next 12 months
# ---------------------------------------------------------------
# The lockdown years were not normal, so we only use data from 2022 onwards.
series = monthly[monthly.index >= "2022-01-01"]


def make_features(dates, t_start):
    # t = how many months have passed, m2..m12 = which month it is (January is the base)
    X = pd.DataFrame({"t": range(t_start, t_start + len(dates))}, index=dates)
    for m in range(2, 13):
        X["m" + str(m)] = (dates.month == m).astype(int)
    return X


X = make_features(series.index, 0)
y = np.log(series.values)        # log is used because growth is in percentage terms

# keep the last 12 months aside for testing
X_train, X_test = X.iloc[:-12], X.iloc[-12:]
y_train = y[:-12]
actual = series.values[-12:]

# Method 1: same as last year (simple baseline)
pred_naive = series.values[-24:-12]

# Method 2: Linear Regression
lin = LinearRegression()
lin.fit(X_train, y_train)
pred_lin = np.exp(lin.predict(X_test))

# Method 3: Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=1)
rf.fit(X_train, y_train)
pred_rf = np.exp(rf.predict(X_test))


def mape(actual, pred):
    return np.mean(np.abs(actual - pred) / actual) * 100


errors = {
    "Same as last year": mape(actual, pred_naive),
    "Linear Regression": mape(actual, pred_lin),
    "Random Forest": mape(actual, pred_rf),
}
print("\nForecast test on the last 12 months (average % error):")
for name in errors:
    print("  ", name, round(errors[name], 2))
best = min(errors, key=errors.get)
print("Best method:", best)

# Chart 7: test result
plt.figure(figsize=(10, 5))
plt.plot(series.index[-12:], actual / 1e6, color="black", linewidth=2.5, label="Actual")
plt.plot(series.index[-12:], pred_naive / 1e6, label="Same as last year (" + str(round(errors["Same as last year"], 1)) + "% error)")
plt.plot(series.index[-12:], pred_lin / 1e6, label="Linear Regression (" + str(round(errors["Linear Regression"], 1)) + "% error)")
plt.plot(series.index[-12:], pred_rf / 1e6, label="Random Forest (" + str(round(errors["Random Forest"], 1)) + "% error)")
plt.title("Forecast test on the last 12 months")
plt.xlabel("Month")
plt.ylabel("Registrations (millions)")
plt.legend()
plt.grid(True)
save_chart("7_forecast_test")

# train the best method on ALL the data and predict the next 12 months
future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=12, freq="MS")
X_future = make_features(future_dates, len(series))
if best == "Same as last year":
    forecast = series.values[-12:]
elif best == "Linear Regression":
    model = LinearRegression().fit(X, y)
    forecast = np.exp(model.predict(X_future))
else:
    model = RandomForestRegressor(n_estimators=200, random_state=1).fit(X, y)
    forecast = np.exp(model.predict(X_future))

last_12 = monthly.iloc[-12:].sum()
next_12 = forecast.sum()
print("\nLast 12 months total:", round(last_12 / 1e6, 2), "million")
print("Next 12 months forecast:", round(next_12 / 1e6, 2), "million (", round((next_12 / last_12 - 1) * 100, 1), "% change )")

# Chart 8: forecast
history = monthly[monthly.index >= last_date - pd.DateOffset(months=35)]
plt.figure(figsize=(10, 5))
plt.plot(history.index, history.values / 1e6, label="Actual")
plt.plot(future_dates, forecast / 1e6, color="red", linestyle="--", marker="o", label="Forecast (" + best + ")")
plt.title("Forecast for the next 12 months")
plt.xlabel("Month")
plt.ylabel("Registrations (millions)")
plt.legend()
plt.grid(True)
save_chart("8_forecast")

# ---------------------------------------------------------------
# Step 4: ML - group similar states with KMeans
# ---------------------------------------------------------------
# for every state: growth, electric share and car share (last 12 months)
state_prev = before.groupby("state")["registrations"].sum()
growth = (state_recent / state_prev - 1) * 100
ev_recent = recent[recent["fuel"] == "Electric"].groupby("state")["registrations"].sum()
ev_pct = ev_recent / state_recent * 100
cars = recent[recent["vehicle_class"] == "Passenger Car"].groupby("state")["registrations"].sum()
car_pct = cars / state_recent * 100

states = pd.DataFrame({"growth": growth, "ev_share": ev_pct, "car_share": car_pct})
scaled = StandardScaler().fit_transform(states)     # put all 3 columns on the same scale

# try 2 and 3 groups and keep the better one (higher silhouette score is better)
best_k = 2
best_score = -1
for k in [2, 3]:
    labels = KMeans(n_clusters=k, n_init=10, random_state=1).fit_predict(scaled)
    score = silhouette_score(scaled, labels)
    print("\nk =", k, "silhouette score =", round(score, 2))
    if score > best_score:
        best_score = score
        best_k = k
states["group"] = KMeans(n_clusters=best_k, n_init=10, random_state=1).fit_predict(scaled) + 1
print("\nUsing k =", best_k)
print(states.round(1).sort_values("group"))
print("\nAverage of each group:")
print(states.groupby("group").mean().round(1))

# Chart 9: groups
plt.figure(figsize=(10, 6))
colors = ["blue", "orange", "green"]
for g in range(1, best_k + 1):
    part = states[states["group"] == g]
    plt.scatter(part["ev_share"], part["growth"], s=100, color=colors[g - 1], label="Group " + str(g))
for name in states.index:
    plt.annotate(name, (states.loc[name, "ev_share"], states.loc[name, "growth"]), xytext=(5, 5), textcoords="offset points")
plt.title("State groups (KMeans)")
plt.xlabel("Electric share (%)")
plt.ylabel("Growth vs previous year (%)")
plt.legend()
plt.grid(True)
save_chart("9_state_groups")

print("\nDone.")
