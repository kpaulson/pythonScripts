def ellipticalIntegralApproximator(k):
    ''' NAME:
       ELLKE

     PURPOSE: 
       Computes Hasting's polynomial approximation for the complete 
       elliptic integral of the first (ek) and second (kk) kind. Combines
       the calculation of both so as not to duplicate the expensive
       calculation of alog10(1-k^2).
       Originally published as supplemental information to Schlawin et al (2010): https://ui.adsabs.harvard.edu/abs/2010ascl.soft11018S/abstract

     INPUTS:

        k - The elliptic modulus.

     OUTPUTS:

        ek - The elliptic integral of the first kind
        kk - The elliptic integral of the second kind

     MODIFICATION HISTORY
     
      2009/04/06 -- Written by Jason Eastman (Ohio State University)
      2022/09/20 -- Adapted to python by Kristoff Paulson
    '''
    import numpy as np

    # This logarithm step below is the computationally expensive part
    m1=1.-k**2
    logm1 = np.log(m1)

    # Constants and function for elliptical integral of the first kind
    a1=0.44325141463
    a2=0.06260601220
    a3=0.04757383546
    a4=0.01736506451
    b1=0.24998368310
    b2=0.09200180037
    b3=0.04069697526
    b4=0.00526449639
    ee1=1.+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ee2=m1*(b1+m1*(b2+m1*(b3+m1*b4)))*(-logm1)
    ek = ee1+ee2
             
    # Constants and function for elliptical integral of the second kind
    a0=1.38629436112
    a1=0.09666344259
    a2=0.03590092383
    a3=0.03742563713
    a4=0.01451196212
    b0=0.5
    b1=0.12498593597
    b2=0.06880248576
    b3=0.03328355346
    b4=0.00441787012
    ek1=a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ek2=(b0+m1*(b1+m1*(b2+m1*(b3+m1*b4))))*logm1
    kk = ek1-ek2

    return(ek, kk)