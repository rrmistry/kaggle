sudo apt install python
sudo apt install python-pip
sudo apt install unzip

# Fix file with kaggle username and password (API Key)
touch ~/.kaggle/kaggle.json

# Install kaggle API
sudo pip install kaggle

# Download competition data
kaggle competitions download -c new-york-city-taxi-fare-prediction

# Unzip large dataset
unzip train.csv.zip

# Copy to Google Storage Account
gsutil cp *.csv gs://BUCKET-NAME/NY_Taxi_Cab/