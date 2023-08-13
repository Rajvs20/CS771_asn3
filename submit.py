import numpy as np
import pickle as pkl
from sklearn.ensemble import RandomForestRegressor
# Define your prediction method here
# We are using the Random Forest mechanism for regression. Two models are being used for O3 and NO2 predictions, and the time is calculated according to the heuristic described in the report.
# df is a dataframe containing timestamps, weather data and potentials
def my_predict( df ):
    rand_for_ozone = pkl.load(open("rand_for_ozone_time.pkl", 'rb'))
    rand_for_no2 = pkl.load(open("rand_for_no2_time.pkl", 'rb'))
    x1 = df["no2op1"].to_numpy()
    x2 = df["no2op2"].to_numpy()
    x3 = df["o3op1"].to_numpy()
    x4 = df["o3op2"].to_numpy()
    x5 = df["temp"].to_numpy()
    x6 = df["humidity"].to_numpy()
    time = df["Time"].to_numpy()
    in_the_moonlight = []
    for i in range(0, time.shape[0]):
        t = time[i][11:]
        if t >= "19:45:00" or t <= "06:55:00": # average sunset and sunrise time
            in_the_moonlight.append(0.0)
        else:
            in_the_moonlight.append(1.0)

    x7 = np.array(in_the_moonlight) # for ozone
    X_1 = []
    for i in range(0, x1.shape[0]):
        x = np.array([x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i]])
        X_1.append(x)
    X = np.array(X_1)
    pred_o3 = rand_for_ozone.predict(X)
    pred_no2 = rand_for_no2.predict(X)
    return ( pred_o3, pred_no2 )
