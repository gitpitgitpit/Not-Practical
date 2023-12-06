import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import norm

df=sb.load_dataset("tips")

x=np.array(df["total_bill"])

x_array=np.linspace(min(x),max(x),100)


x_mean=np.mean(x)
x_std=np.std(x)
pdf=norm.pdf(x_array,x_mean,x_std)
plt.plot(x_array,pdf)
plt.grid()
plt.show()

skew=skew(x,axis=0,bias=True)
print(skew)

kurt=kurtosis(x,axis=0,fisher=False,bias=True)
print(kurt)