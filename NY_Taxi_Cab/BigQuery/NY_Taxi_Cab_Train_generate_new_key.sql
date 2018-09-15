# Generate new key based on pickup time, location and dropoff location

SELECT
CONCAT(CAST(pickup_datetime AS STRING), CAST(pickuplon AS STRING), CAST(pickuplat AS STRING), CAST(dropofflon AS STRING), CAST(dropofflat AS STRING)) AS key,
key AS key_original,
fare_amount,
pickup_datetime,
dayofweek,
hourofday,
pickuplon,
pickuplat,
dropofflon,
dropofflat,
passengers
FROM
NY_Taxi_Cab.train_filter_stage2