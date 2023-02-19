
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from utils import plot_lat_long, euc_distance, predict_random
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    df = pd.read_csv('NYC_taxi.csv', parse_dates=['pickup_datetime'], nrows=500000)

    # range of longitude for NYC
    nyc_min_longitude = -74.05
    nyc_max_longitude = -73.75
    # range of latitude for NYC
    nyc_min_latitude = 40.63
    nyc_max_latitude = 40.85
    df2 = df.copy(deep=True)
    for long in ['pickup_longitude', 'dropoff_longitude']:
        df2 = df2[(df2[long] > nyc_min_longitude) & (df2[long] <
                                                     nyc_max_longitude)]
    for lat in ['pickup_latitude', 'dropoff_latitude']:
        df2 = df2[(df2[lat] > nyc_min_latitude) & (df2[lat] <
                                                   nyc_max_latitude)]

    landmarks = {'JFK Airport': (-73.78, 40.643),
                 'Laguardia Airport': (-73.87, 40.77),
                 'Midtown': (-73.98, 40.76),
                 'Lower Manhattan': (-74.00, 40.72),
                 'Upper Manhattan': (-73.94, 40.82),
                 'Brooklyn': (-73.95, 40.66)}

    print(df.head())
    plot_lat_long(df2, landmarks, points='Pickup')
    # plot_lat_long(df2, landmarks, points='Drop Off')

    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['hour'] = df['pickup_datetime'].dt.hour

    df['hour'].plot.hist(bins=24, ec='black')
    plt.title('Pickup Hour Histogram')
    plt.xlabel('Hour')
    plt.show()
    print(df.isnull().sum())
    df = df.dropna()
    print(df.describe())

    df['fare_amount'].hist(bins=500)
    plt.xlabel("Fare")
    plt.title("Histogram of Fares")
    plt.show()

    df = df[(df['fare_amount'] >= 0) & (df['fare_amount'] <= 100)]
    df['passenger_count'].hist(bins=6, ec='black')
    plt.xlabel("Passenger Count")
    plt.title("Histogram of Passenger Count")
    plt.show()

    df.loc[df['passenger_count'] == 0, 'passenger_count'] = 1

    df.plot.scatter('pickup_longitude', 'pickup_latitude')
    plt.show()

    # range of longitude for NYC
    nyc_min_longitude = -74.05
    nyc_max_longitude = -73.75
    # range of latitude for NYC
    nyc_min_latitude = 40.63
    nyc_max_latitude = 40.85
    # only consider locations within NYC
    for long in ['pickup_longitude', 'dropoff_longitude']:
        df = df[(df[long] > nyc_min_longitude) & (df[long] <
                                                  nyc_max_longitude)]
    for lat in ['pickup_latitude', 'dropoff_latitude']:
        df = df[(df[lat] > nyc_min_latitude) & (df[lat] < nyc_max_latitude)]

    print(df.head()['pickup_datetime'])

    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['hour'] = df['pickup_datetime'].dt.hour

    print(df.loc[:5, ['pickup_datetime', 'year', 'month',
                      'day', 'day_of_week', 'hour']])

    df = df.drop(['pickup_datetime'], axis=1)

    df['distance'] = euc_distance(df['pickup_latitude'],
                                  df['pickup_longitude'],
                                  df['dropoff_latitude'],
                                  df['dropoff_longitude'])

    df.plot.scatter('fare_amount', 'distance')

    airports = {'JFK_Airport': (-73.78, 40.643),
                'Laguardia_Airport': (-73.87, 40.77),
                'Newark_Airport': (-74.18, 40.69)}
    for airport in airports:
        df['pickup_dist_' + airport] = euc_distance(df['pickup_latitude'],
                                                    df['pickup_longitude'],
                                                    airports[airport][1],
                                                    airports[airport][0])
        df['dropoff_dist_' + airport] = euc_distance(df['dropoff_latitude'],
                                                     df['dropoff_longitude'],
                                                     airports[airport][1],
                                                     airports[airport][0])

    print(df[['key', 'pickup_longitude', 'pickup_latitude',
              'dropoff_longitude', 'dropoff_latitude',
              'pickup_dist_JFK_Airport',
              'dropoff_dist_JFK_Airport']].head())

    df = df.drop(['key'], axis=1)

    df_prescaled = df.copy()
    df_scaled = df.drop(['fare_amount'], axis=1)

    df_scaled = scale(df_scaled)

    cols = df.columns.tolist()
    cols.remove('fare_amount')
    df_scaled = pd.DataFrame(df_scaled, columns=cols, index=df.index)
    df_scaled = pd.concat([df_scaled, df['fare_amount']], axis=1)
    df = df_scaled.copy()
    print(df)

    # implement
    X = df.loc[:, df.columns != 'fare_amount']
    y = df.loc[:, 'fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))

    model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.fit(X_train, y_train, epochs=1)

    predict_random(df_prescaled, X_test, model)

    train_pred = model.predict(X_train)
    print(train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    print("Train RMSE: {:0.2f}".format(train_rmse))
    print("Test RMSE: {:0.2f}".format(test_rmse))
    # plt.show()
