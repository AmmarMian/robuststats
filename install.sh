# create conda environment
conda create -n robuststats python=3.9 --yes 
eval "$(conda shell.bash hook)"  # bug fix: https://github.com/conda/conda/issues/7980#issuecomment-492784093 
conda activate robuststats

# Dependencies
conda install python ipython --yes
pip install -y --file requirements.txt

python setup.py install
