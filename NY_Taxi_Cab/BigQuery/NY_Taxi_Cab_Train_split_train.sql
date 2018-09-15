# Split train dataset into train, valid, test
# HASH the key, mod with 10
# - Train => hash < 7
# - Test  => hash = 7
# - Valid => hash > 7

SELECT
  key,
  key_original,
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
  NY_Taxi_Cab.train_generate_new_key
WHERE
  MOD(ABS(FARM_FINGERPRINT(key)), 10) < 7