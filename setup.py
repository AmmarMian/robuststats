from setuptools import setup, find_packages

setup(name='robuststats',
      version='0',
      description='Robust Statistics toolbox',
      url='',
      author='Ammar Mian',
      author_email='ammar.mian@protonmail.com',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'scikit-learn',  'joblib', 'autograd', 'pymanopt', 'geomstats', 'tqdm'],
      zip_safe=False)
