from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def func(t, v, k):
    """ computes the function S(t) with constants v and k """
    
    # TODO: return the given function S(t)
    return v*(t-((1-np.exp(-k*t))/k))
    # END TODO


def find_constants(df: pd.DataFrame, func: Callable):
    """ returns the constants v and k """

    v = 0
    k = 0

    # TODO: fit a curve using SciPy to estimate v and k
    (v,k),_=curve_fit(func,df['t'].to_numpy(),df['S'].to_numpy())
    # END TODO

    return v, k


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    v, k = find_constants(df, func)
    v = v.round(4)
    k = k.round(4)
    print(v, k)

    # TODO: plot a histogram and save to fit_curve.png
    x=np.linspace(0,0.5,10000)
    
    plt.plot(x,func(x,v,k),'r',label=f'fit: v={v},k={k}')
    plt.scatter(df['t'],df['S'],marker='*',color='blue',label='data')
    plt.xlabel('t')
    plt.ylabel('S')
    plt.legend()
    plt.savefig("fit_curve.png")
    # END TODO
