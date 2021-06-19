# create conda environment
conda create -n robust_stats --yes 
eval "$(conda shell.bash hook)"  # bug fix: https://github.com/conda/conda/issues/7980#issuecomment-492784093 
conda activate robust_stats

# Dependencies
conda config --add channels conda-forge
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
