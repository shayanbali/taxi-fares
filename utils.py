import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_lat_long(df, landmarks, points='Pickup'):
    plt.figure(figsize=(12, 12))  # set figure size
    if points == 'pickup':
        plt.plot(list(df.pickup_longitude), list(df.pickup_latitude),
                 '.', markersize=1)
    else:
        plt.plot(list(df.dropoff_longitude), list(df.dropoff_latitude),
                 '.', markersize=1)
    for landmark in landmarks:
        plt.plot(landmarks[landmark][0], landmarks[landmark][1],
                 '*', markersize=15, alpha=1, color='r')
        plt.annotate(landmark, (landmarks[landmark][0] + 0.005,
                                landmarks[landmark][1] + 0.005), color='r',
                     backgroundcolor='w')
    plt.title("{} Locations in NYC Illustrated".format(points))
    plt.grid(None)
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.show()


def preprocess(df):
    # remove missing values in the dataframe
    def remove_missing_values(df):
        df = df.dropna()
        return df

    # remove outliers in fare amount
    def remove_fare_amount_outliers(df, lower_bound, upper_bound):
        df = df[(df['fare_amount'] >= lower_bound) & (df['fare_amount'] <= upper_bound)]
        return df

        # replace outliers in passenger count with the mode
        def replace_passenger_count_outliers(df):
            mode = df['passenger_count'].mode()
            df.loc[df['passenger_count'] == 0, 'passenger_count'] = mode
            return df

        # remove outliers in latitude and longitude
        def remove_lat_long_outliers(df):
            # range of longitude for NYC
            nyc_min_longitude = -74.05
            nyc_max_longitude = -73.75
            # range of latitude for NYC
            nyc_min_latitude = 40.63
            nyc_max_latitude = 40.85
            # only consider locations within New York City
            for long in ['pickup_longitude', 'dropoff_longitude']:
                df = df[(df[long] > nyc_min_longitude) &
                        (df[long] < nyc_max_longitude)]
            for lat in ['pickup_latitude', 'dropoff_latitude']:
                df = df[(df[lat] > nyc_min_latitude) &
                        (df[lat] < nyc_max_latitude)]
            return df

        df = remove_missing_values(df)
        df = remove_fare_amount_outliers(df, lower_bound=0,
                                         upper_bound=100)
        df = replace_passenger_count_outliers(df)
        df = remove_lat_long_outliers(df)
        return df


def euc_distance(lat1, long1, lat2, long2):
    return ((lat1 - lat2) ** 2 + (long1 - long2) ** 2) ** 0.5


def feature_engineer(df):
    # create new columns for year, month, day, day of week and hour
    def create_time_features(df):
        df['year'] = df['pickup_datetime'].dt.year
        df['month'] = df['pickup_datetime'].dt.month
        df['day'] = df['pickup_datetime'].dt.day
        df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
        df['hour'] = df['pickup_datetime'].dt.hour
        df = df.drop(['pickup_datetime'], axis=1)
        return df

    # function to calculate euclidean distance
    def euc_distance(lat1, long1, lat2, long2):
        return (((lat1 - lat2) ** 2 + (long1 - long2) ** 2) ** 0.5)

    # create new column for the distance travelled
    def create_pickup_dropoff_dist_features(df):
        df['travel_distance'] = euc_distance(df['pickup_latitude'],
                                             df['pickup_longitude'],
                                             df['dropoff_latitude'],
                                             df['dropoff_longitude'])
        return df

    # create new column for the distance away from airports
    def create_airport_dist_features(df):
        airports = {'JFK_Airport': (-73.78, 40.643),
                    'Laguardia_Airport': (-73.87, 40.77),
                    'Newark_Airport': (-74.18, 40.69)}
        for k in airports:
            df['pickup_dist_' + k] = euc_distance(df['pickup_latitude'],
                                                  df['pickup_longitude'],
                                                  airports[k][1],
                                                  airports[k][0])
            df['dropoff_dist_' + k] = euc_distance(df['dropoff_latitude'],
                                                   df['dropoff_longitude'],
                                                   airports[k][1],
                                                   airports[k][0])
        return df

    df = create_time_features(df)
    df = create_pickup_dropoff_dist_features(df)
    df = create_airport_dist_features(df)
    df = df.drop(['key'], axis=1)
    return df


def predict_random(df_prescaled, X_test, model):
    sample = X_test.sample(n=1, random_state=np.random.randint(low=0,
                                                               high=10000))
    idx = sample.index[0]
    actual_fare = df_prescaled.loc[idx, 'fare_amount']
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                 'Saturday', 'Sunday']
    day_of_week = day_names[df_prescaled.loc[idx, 'day_of_week']]
    hour = df_prescaled.loc[idx, 'hour']
    predicted_fare = model.predict(sample)[0][0]
    rmse = np.sqrt(np.square(predicted_fare - actual_fare))
    print("Trip Details: {}, {}:00hrs".format(day_of_week, hour))
    print("Actual fare: ${:0.2f}".format(actual_fare))
    print("Predicted fare: ${:0.2f}".format(predicted_fare))
    print("RMSE: ${:0.2f}".format(rmse))
