'''
File: robust_estimation_real.py
File Created: Friday, 9th July 2021 10:07:35 am
Author: Ammar Mian (ammar.mian@univ-smb.fr)
-----
Last Modified: Tuesday, 21st December 2021 5:40:06 pm
Modified By: Ammar Mian (ammar.mian@univ-smb.fr>)
-----
Examples of covariance estimation with various estimators
Real-valued case
-----
Copyright 2021, Université Savoie Mont-Blanc
'''

import numpy as np
from robuststats.estimation.covariance import TylerShapeMatrix, \
    get_normalisation_function, CenteredStudentMLE, HuberMEstimator
from robuststats.utils.linalg import ToeplitzMatrix
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import multivariate_t


import logging
logging.basicConfig(level='INFO')

if __name__ == '__main__':

    # Deterministic execution
    base_seed = 7777
    np.random.seed(base_seed)

    # Data parameters
    n_features = 20
    n_samples = 80

    print("Generating data")
    df = 3
    normalisation = "trace"
    S = get_normalisation_function(normalisation)
    covariance = ToeplitzMatrix(0.97, n_features, dtype=float)
    # covariance = covariance / S(covariance)

    model = multivariate_t(shape=covariance, df=df)
    X = model.rvs(size=n_samples)
    
    print("Performing Sample covariance matrix estimation")
    estimator_scm = EmpiricalCovariance(assume_centered=True)
    estimator_scm.fit(X)
    
    print("Performing Tyler fixed-point estimation of covariance matrix")
    estimator_tyler = TylerShapeMatrix(normalisation=normalisation, verbosity=True)
    estimator_tyler.fit(X)
    
    print("Performing Student-t MLE estimation of covariance matrix")
    estimator_student = CenteredStudentMLE(df=3, verbosity=True, iter_max=1000)
    estimator_student.fit(X)
    
    print("Performing Huber's M-estimation of covariance matrix")
    estimator_huber = HuberMEstimator(q=0.1, verbosity=True, iter_max=1000)
    estimator_huber.fit(X)

    backend = "matplotlib"
    if backend=="matplotlib":
        import matplotlib.pyplot as plt
        print("Saving plots to Robust_shape_estimation.png")
        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        im = axes[0].imshow(covariance, aspect='auto')
        axes[0].set_title("True Covariance")
        fig.colorbar(im, ax=axes[0])

        im = axes[1].imshow(estimator_tyler.covariance_, aspect='auto')
        axes[1].set_title(f"Estimated Covariance with Tyler $N={n_samples}$")
        fig.colorbar(im, ax=axes[1])
        
        im = axes[2].imshow(estimator_student.covariance_, aspect='auto')
        axes[2].set_title(f"Estimated Covariance with Student-t MLE $N={n_samples}$")
        fig.colorbar(im, ax=axes[2])
        
        im = axes[3].imshow(estimator_huber.covariance_, aspect='auto')
        axes[3].set_title(f"Estimated Covariance with Huber's M-estimator $N={n_samples}$")
        fig.colorbar(im, ax=axes[3])
        
        im = axes[4].imshow(estimator_scm.covariance_, aspect='auto')
        axes[4].set_title(f"Estimated Covariance with SCM $N={n_samples}$")
        fig.colorbar(im, ax=axes[4])
        
        # plt.show()
        plt.savefig("Robust_shape_estimation.png")

    elif backend=="plotly":
        import plotly.express as px
        print("Writing html export of results")
        fig = px.imshow(covariance)
        fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        fig.write_html("true_covariance.html")
        
        fig = px.imshow(estimator_scm.covariance_)
        fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        fig.write_html("scm_covariance.html")
        
        fig = px.imshow(estimator_tyler.covariance_)
        fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        fig.write_html("tyler_covariance.html")
        
        fig = px.imshow(estimator_student.covariance_)
        fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        fig.write_html("student_mle_covariance.html")
        
        fig = px.imshow(estimator_huber.covariance_)
        fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        fig.write_html("huber_covariance.html")

    else:
        import pandas as pd
        print("Saving data to csv for later use")
        pd.DataFrame(covariance).to_csv("true_covariance.csv")
        np.savetxt("tyler.csv", estimator_tyler.covariance_, delimiter=",")
        np.savetxt("scm.csv", estimator_scm.covariance_, delimiter=",")
        np.savetxt("student_mle.csv", estimator_student.covariance_, delimiter=",")
        np.savetxt("huber.csv", estimator_huber.covariance_, delimiter=",")
