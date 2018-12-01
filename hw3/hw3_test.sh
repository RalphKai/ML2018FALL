wget https://www.dropbox.com/s/e4jnxvw88v61k8m/model1.hdf5 -O model1.hdf5
wget https://www.dropbox.com/s/t9jqbajxiha8r08/model2.hdf5 -O model2.hdf5
wget https://www.dropbox.com/s/g85ufpwyrrps8r4/model3.hdf5 -O model3.hdf5
python3 test.py $1 $2
