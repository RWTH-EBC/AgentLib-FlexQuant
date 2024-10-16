import pandas as pd
import numpy as np

df = pd.read_csv("price_ele_2015_day_ahead.csv", delimiter=",")
df['PowerPrice'] = df['PowerPrice'].ffill()
power_prices_base = []
power_prices_peak = []

for i, hour_value in enumerate(df['Datum']):
    if 7 < hour_value < 20:
        power_prices_peak.append(df['PowerPrice'].iloc[i])
    else:
        power_prices_base.append(df['PowerPrice'].iloc[i])

# EPEX price in MWh
power_price_peak_mean = np.mean(power_prices_peak)
power_price_base_mean = np.mean(power_prices_base)

power_price_median = df['PowerPrice'].median()
power_price_mean = df['PowerPrice'].mean()
peak_median = np.median(power_prices_peak)
base_median = np.median(power_prices_base)

def calc_market_price(price):
    """
    Probably not optimal for Peak and Base Model
    Taxes overweight the Peak and Base Difference
    """
    new_price = price*0.1*1.1889 + 21.914  # approx from tibber values
    return new_price

# Local market price in kWh
power_price_peak_after_tax = calc_market_price(power_price_peak_mean)
power_price_base_after_tax = calc_market_price(power_price_base_mean)
power_price_after_tax = calc_market_price(power_price_median)

# Peak and Base diffence between 15 to 20%:	https://energysales.vattenfall.de/publikationen/artikel/base-vs-peak-die-differenz-waechst

power_price_base = power_price_after_tax * 0.9
power_price_peak = power_price_after_tax * 1.1
const_power_price = 3.018*0.1*1.1889 + 21.914

print(power_price_peak, power_price_base)