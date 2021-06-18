# create conda environment
conda create -n robuststats python=3.7 --yes 
eval "$(conda shell.bash hook)"  # bug fix: https://github.com/conda/conda/issues/7980#issuecomment-492784093 
conda activate robuststats

# install pymanopt fork
cd submodules/pymanopt_fork
pip install -r requirements.txt
python setup.py install

# install pyCovariance
cd ../pyCovariance
pip install -r requirements.txt
python setup.py install

# install robuststats
cd ../..
python setup.py install

# Other useful packages for examples
conda install ipython
conda install plotly
conda install jupyter
