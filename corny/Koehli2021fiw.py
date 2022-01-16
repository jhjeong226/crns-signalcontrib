import numpy as np

def sm2N(sm, N0, off=0.02, bd=1, a0=0.0808, a1=0.115, a2=0.372):
    return(N0*(0.0808/(sm/bd+0.115+off)+0.372))

def sm2N_Koehli(sm, h=9, off=0.02, bd=1, func='vers1', method=None, bio=0):
    #vers1: Sep25_responsef, Sep25_Ewindow, vers2: Jan23_uranos, Jan23_mcnpfull, Mar12_atmprof

    # total sm
    smt = sm + off
    smt *= 1.43/bd
    if smt == 0.0:
        smt = 0.001
    # nothing to do with bd
    p = []

    ################# PLEASE DOUBLE-CHECK THE FUNCTIONS AND VALUES ##################

    if func == 'vers1':
        if method == 'Sep25_responsef':          p = [4.179, 0.0230, 0.200, 1.436, 0.902, -0.00308, -0.0716, -0.0000163, 0.00164]
        elif method == 'Sep25_Ewindow':          p = [8.284, 0.0191, 0.143, 2.384, 0.760, -0.00344, -0.1310, -0.0000240, 0.00280]

        N = (p[1]+p[2]*smt)/(smt+p[1])*(p[0]+p[6]*h +p[8]* h**2+p[7]*h**3) + np.exp(-p[3]*smt)*(p[4]+p[5]*h)

    elif func == 'vers2':
        if method == 'Jan23_uranos':             p = [4.2580, 0.0212, 0.206, 1.776, 0.241, -0.00058, -0.02800, 0.0003200, -0.0000000180]
        elif method == 'Jan23_mcnpfull':         p = [7.0000, 0.0250, 0.233, 4.325, 0.156, -0.00066, -0.01200, 0.0004100, -0.0000000410]
        elif method == 'Mar12_atmprof':          p = [4.4775, 0.0230, 0.217, 1.540, 0.213, -0.00022, -0.03800, 0.0003100, -0.0000000003]

        elif method == 'Mar21_mcnp_drf':         p = [1.0940, 0.0280, 0.254, 3.537, 0.139, -0.00140, -0.00880, 0.0001150,  0.0000000000]
        elif method == 'Mar21_mcnp_ewin':        p = [1.2650, 0.0259, 0.135, 1.237, 0.063, -0.00021, -0.01170, 0.0001200,  0.0000000000]
        elif method == 'Mar21_uranos_drf':       p = [1.0240, 0.0226, 0.207, 1.625, 0.235, -0.00290, -0.00930, 0.0000740,  0.0000000000]
        elif method == 'Mar21_uranos_ewin':      p = [1.2230, 0.0185, 0.142, 2.568, 0.155, -0.00047, -0.01190, 0.0000920,  0.0000000000]

        elif method == 'Mar22_mcnp_drf_Jan':     p = [1.0820, 0.0250, 0.235, 4.360, 0.156, -0.00071, -0.00610, 0.0000500,  0.0000000000]
        elif method == 'Mar22_mcnp_ewin_gd':     p = [1.1630, 0.0244, 0.182, 4.358, 0.118, -0.00046, -0.00747, 0.0000580,  0.0000000000]
        elif method == 'Mar22_uranos_drf_gd':    p = [1.1180, 0.0221, 0.173, 2.300, 0.184, -0.00064, -0.01000, 0.0000810,  0.0000000000]
        elif method == 'Mar22_uranos_ewin_chi2': p = [1.0220, 0.0218, 0.199, 1.647, 0.243, -0.00029, -0.00960, 0.0000780,  0.0000000000]
        elif method == 'Mar22_uranos_drf_h200m': p = [1.0210, 0.0222, 0.203, 1.600, 0.244, -0.00061, -0.00930, 0.0000740,  0.0000000000]

        elif method == 'Aug08_mcnp_drf':         p = [1.110773444917129, 0.034319446894963, 0.180046592985848, 1.211393214064259, 0.093433803170610, -1.877788035e-005, -0.00698637546803, 5.0316941885e-005, 0.0000000000]
        elif method == 'Aug08_mcnp_ewin':        p = [1.271225645585415, 0.024790265564895, 0.107603498535911, 1.243101823658557, 0.057146624195463, -1.93729201894976, -0.00866217333051, 6.198559205414182, 0.0000000000]
        elif method == 'Aug12_uranos_drf':       p = [1.042588152355816, 0.024362250648228, 0.222359434641456, 1.791314246517330, 0.197766380530824, -0.00053814104957, -0.00820189794785, 6.6412111902e-005, 0.0000000000]
        elif method == 'Aug12_uranos_ewin':      p = [1.209060105287452, 0.021546879683024, 0.129925023764294, 1.872444149093526, 0.128883139550384, -0.00047134595878, -0.01080226893400, 8.8939419535e-005, 0.0000000000]
        elif method == 'Aug13_uranos_atmprof':   p = [1.044276170094123, 0.024099232055379, 0.227317847739138, 1.782905159416135, 0.198949609723093, -0.00059182327737, -0.00897372356601, 7.3282344356e-005, 0.0000000000]
        elif method == 'Aug13_uranos_atmprof2':  p = [4.31237,           0.020765,          0.21020,           1.87120,           0.16341,           -0.00052,          -0.00225,          0.000308,         -1.9639e-8]

        N = (p[1]+p[2]*smt)/(smt+p[1])*(p[0]+p[6]*h+p[7]*h**2+p[8]*h**3/smt)+np.exp(-p[3]*smt)*(p[4]+p[5]*(h + bio/5*1000))

    return(N)#/N.mean())


def Calibrate_N0_Desilets(N, sm, bd=1, lw=0, owe=0, a0=0.0808, a1=0.372, a2=0.115):
    return(N/(a0 / (sm/bd + a2 + lw + owe) + a1))

def N2SM_Desilets(N, N0, bd=1, lw=0, owe=0, a0=0.0808, a1=0.372, a2=0.115):
    return((a0/(N/int(N0)-a1)-a2 -lw - owe) * bd)

def N2SM_Schmidt_single(N, hum, bd=1, lw=0, owe=0, method='Mar21_uranos_drf'):
    t0 = 0.0
    t1 = 1.0
    n0 = sm2N_Koehli(0.0, hum, method=method, func='vers2', off=lw+owe, bd=bd)
    n1 = sm2N_Koehli(1.0, hum, method=method, func='vers2', off=lw+owe, bd=bd)
    while t1 - t0 > 0.0001:
        t2 = 0.5*(t0+t1);
        n2 = sm2N_Koehli(t2, hum, method=method, func='vers2', off=lw+owe, bd=bd)
        if N < n2:
            t0 = t2
            n0 = n2
        else:
            t1 = t2
            n1 = n2
    t2 = 0.5*(t0+t1)
    return(t2)

def N2SM_Schmidt(data, Nstr, humstr, N0, bdstr='bd', lwstr='lw', owestr='owe', method='Mar21_uranos_drf'):
    sm = data.apply((lambda x: N2SM_Schmidt_single(x[Nstr]/N0*0.77, x[humstr], lw=x[lwstr], owe=x[owestr], bd=x[bdstr], method=method)), axis=1)
    return(sm)

