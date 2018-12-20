wget https://www.dropbox.com/s/fo649lun6m8y4xn/model.hdf5?dl=1 model.hdf5
wget https://www.dropbox.com/s/os4hupwsuvp3r2i/model_1.hdf5?dl=1 model_1.hdf5
wget https://www.dropbox.com/s/dkdu4g70x3pwooj/model_2.hdf5?dl=1 model_2.hdf5
python test.py $1 $2 $3
