mkdir data
cd data

curl -o non-vehicles_smallset.zip https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles_smallset.zip 
curl -o vehicles_smallset.zip https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles_smallset.zip
unzip non-vehicles_smallset.zip
unzip vehicles_smallset.zip

curl -o vehicles.zip https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
curl -o non-vehicles.zip https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip
unzip vehicles.zip
unzip non-vehicles.zip

# These are 1.3 and 3 GB 
curl -L -o object-detection.tar.gz http://bit.ly/udacity-annoations-crowdai
curl -L -o object-dataset.tar.gz http://bit.ly/udacity-annotations-autti