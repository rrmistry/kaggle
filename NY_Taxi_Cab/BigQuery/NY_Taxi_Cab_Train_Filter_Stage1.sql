--// Validate Bounds: http://boundingbox.klokantech.com/

WITH daynames AS
  (SELECT ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'] AS daysofweek)
SELECT
  key,
  fare_amount,
  pickup_datetime,
  daysofweek[ORDINAL(EXTRACT(DAYOFWEEK FROM pickup_datetime))] AS dayofweek,
  EXTRACT(HOUR FROM pickup_datetime) AS hourofday,
  pickup_longitude AS pickuplon,
  pickup_latitude AS pickuplat,
  dropoff_longitude AS dropofflon,
  dropoff_latitude AS dropofflat,
  passenger_count AS passengers,
  ((pickup_longitude-dropoff_longitude)*(pickup_longitude-dropoff_longitude) + (pickup_latitude-dropoff_latitude)*(pickup_latitude-dropoff_latitude)) AS trip_distance_sqr
FROM
  NY_Taxi_Cab.train, daynames
WHERE fare_amount >= 2.5
  AND pickup_longitude > -78
  AND pickup_longitude < -70
  AND dropoff_longitude > -78
  AND dropoff_longitude < -70
  AND pickup_latitude > 37
  AND pickup_latitude < 45
  AND dropoff_latitude > 37
  AND dropoff_latitude < 45
  AND passenger_count > 0