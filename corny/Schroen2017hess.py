"""
CoRNy CORN
    Functions specific for data processing of COsmic-Ray Neutrons
    from: https://git.ufz.de/CRNS/cornish_pasdy/-/blob/master/corny/corn.py
    version: 0.62
"""
import pandas
import pandas as pd
import numpy as np
import os

# Footprint

def D86(sm, bd=1, r=1):
    return(1/bd*( 8.321+0.14249*( 0.96655+np.exp(-r/100))*(20.0+sm) / (0.0429+sm)))

def Weight_d(d, D):
    return(np.exp(-2*d/D))

# Preload footprint radius matrix
preload_footprint_data = np.loadtxt('corny/footprint_radius.csv')


def get_footprint(sm, h, p, lookup_file=None):
    """
    usage: data['footprint_radius'] = data.apply(lambda row: get_footprint( row[smv], row[ah], row[p] ), axis=1)
    """
    if np.isnan(sm) or np.isnan(h) or np.isnan(p): return(np.nan)
    if sm <  0.01: sm =  0.01
    if sm >  0.49: sm =  0.49
    if h  > 30   : h  = 30
    #print(sm, h, p, int(round(100*sm)), int(round(h)))
    if lookup_file is None:
        footprint_data = preload_footprint_data
        #footprint_data = np.loadtxt(os.path.join(pasdy_path ,'corny/footprint_radius.csv'))
    elif isinstance(lookup_file, str):
        if os.path.exists(lookup_file):
            footprint_data = np.loadtxt(lookup_file)
        else:
            return(np.nan)
    elif isinstance(lookup_file, np.ndarray):
        footprint_data = lookup_file
    else:
        return(np.nan)
    return(footprint_data[int(round(100*sm))][int(round(h))] * 0.4922/(0.86-np.exp(-p/1013.25)))


def get_footprint_volume(depth, radius, theta, bd):
    return((depth + D86(theta, bd, radius))*0.01*0.47*radius**2*3.141 /1000)

    # 0.44 (dry) ..0.5 (wet) is roughly the average D over radii

def Wr_approx(r=1):
    return((30*np.exp(-r/1.6)+np.exp(-r/100))*(1-np.exp(-3.7*r)))

def Wr(r=1, sm=0.1, hum=5, normalize=False):
    x = hum
    y = sm
    a00 = 8735; a01 = 22.689; a02 = 11720; a03 = 0.00978; a04 = 9306; a05 = 0.003632
    a10 = 2.7925e-002; a11 = 6.6577; a12 = 0.028544; a13 = 0.002455; a14 = 6.851e-005; a15 = 12.2755
    a20 = 247970; a21 = 23.289; a22 = 374655; a23 = 0.00191; a24 = 258552
    a30 = 5.4818e-002; a31 = 21.032; a32 = 0.6373; a33 = 0.0791; a34 = 5.425e-004
    b00 = 39006; b01 = 15002337; b02 = 2009.24; b03 = 0.01181; b04 = 3.146; b05 = 16.7417; b06 = 3727
    b10 = 6.031e-005; b11 = 98.5; b12 = 0.0013826
    b20 = 11747; b21 = 55.033; b22 = 4521; b23 = 0.01998; b24 = 0.00604; b25 = 3347.4; b26 = 0.00475
    b30 = 1.543e-002; b31 = 13.29; b32 = 1.807e-002; b33 = 0.0011; b34 = 8.81e-005; b35 = 0.0405; b36 = 26.74
    A0 = (a00*(1+a03*x)*np.exp(-a01*y)+a02*(1+a05*x)-a04*y)
    A1 = ((-a10+a14*x)*np.exp(-a11*y/(1+a15*y))+a12)*(1+x*a13)
    A2 = (a20*(1+a23*x)*np.exp(-a21*y)+a22-a24*y)
    A3 = a30*np.exp(-a31*y)+a32-a33*y+a34*x
    B0 = (b00-b01/(b02*y+x-0.13))*(b03-y)*np.exp(-b04*y)-b05*x*y+b06
    B1 = b10*(x+b11)+b12*y
    B2 = (b20*(1-b26*x)*np.exp(-b21*y*(1-x*b24))+b22-b25*y)*(2+x*b23)
    B3 = ((-b30+b34*x)*np.exp(-b31*y/(1+b35*x+b36*y))+b32)*(2+x*b33)
    
    if np.isscalar(r):
        if r <= 1:               w = (A0*(np.exp(-A1*r)) + A2*np.exp(-A3*r))*(1-np.exp(-3.7*r))
        elif (r > 1) & (r < 50): w =  A0*(np.exp(-A1*r)) + A2*np.exp(-A3*r)
        elif (r >= 50):          w =  B0*(np.exp(-B1*r)) + B2*np.exp(-B3*r)
        return(w)
    else:
        W = pandas.DataFrame()
        W['r'] = r
        W['w'] = 0
        W.loc[W.r <=  1,'w'] = (A0*(np.exp(-A1*W.loc[W.r <=  1,'r'])) + A2*np.exp(-A3*W.loc[W.r <=  1,'r']))*(1-np.exp(-3.7*W.loc[W.r <=   1,'r']))
        W.loc[W.r >   1,'w'] =  A0*(np.exp(-A1*W.loc[W.r >   1,'r'])) + A2*np.exp(-A3*W.loc[W.r >   1,'r'])
        W.loc[W.r >= 50,'w'] =  B0*(np.exp(-B1*W.loc[W.r >= 50,'r'])) + B2*np.exp(-B3*W.loc[W.r >= 50,'r'])
        if normalize:
            W.w /= W.w.sum()
        return(W.w.values)
