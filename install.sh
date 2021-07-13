# create conda environment
conda create -n robust_stats python=3.7 --yes 
eval "$(conda shell.bash hook)"  # bug fix: https://github.com/conda/conda/issues/7980#issuecomment-492784093 
conda activate robust_stats

# Dependencies
conda config --add channels conda-forge
conda install -y --file requirements.txt

# cloning submodules
git submodule update --init --recursive

# install pymanopt fork
cd submodules/pymanopt
python setup.py install

# install pyCovariance
cd ../pyCovariance
python setup.py install

# install robuststats
cd ../..
python setup.py install
