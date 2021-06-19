# create conda environment
conda create -n robuststats --yes 
eval "$(conda shell.bash hook)"  # bug fix: https://github.com/conda/conda/issues/7980#issuecomment-492784093 
conda activate robuststats

# Dependencies
conda install -y --file requirements.txt

# install pymanopt fork
cd submodules/pymanopt_fork
python setup.py install

# install pyCovariance
cd ../pyCovariance
python setup.py install

# install robuststats
cd ../..
python setup.py install
