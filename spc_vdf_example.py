
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

def ellipticalSPCVDF_MikeStyle(totalFlux, mv_lo, mv_hi):
    '''
    Mike's conversion of SPC L2 differential_flux_density to Velocity Distribution Function units
    (Still working on why it's not the same as above, presumably units somewhere)
    '''
    import numpy as np
    
    # Constants
    # express V(t) in the form V(t) = A*(1 - m*sin(omega*t)^2)
    # and express v(t) in the form v(t) = c*sqrt(V(t)) = c*sqrt(A)*sqrt( 1 - m*sin(omega*t)^2 )
    # where c = sqrt(2*e0/mp)
    c = 13.84177                  # [km s^-1 V^1/2]
    
    # effective width of an energy interval dV
    dV = abs(mv_hi - mv_lo)# / 2.
        
    # center voltage V
    V = (mv_hi + mv_lo) / 2.
    
    A = mv_hi
    m = dV / A


    #the average speed, vkms_mid, is given by the complete elliptical integral of the 2nd kind (though Mike's notes use the first kind from Hasting's approximation):
    #<v(t)> = c*sqrt(A)*E(m)/(pi/2)
    #The following calls the polynomial approximation (Hasting's Approximation)
    (ellipticalFirst, ellipticalSecond) = ellipticalIntegralApproximator( sqrt(m) )
    
    vkms_mid = c * np.sqrt(A) * ellipticalFirst / (np.pi/2.)
    v_z = vkms_mid #Just converting Mike's naming convention to mine
    
    #v_z.putProperty(QDataSet.NAME,'v_z')    # This "putProperty" stuff is an autoplot thing. Ignore it here.
    #v_z.putProperty(QDataSet.LABEL,'v!Bz!N (km/s)')
    
  
    #the RMS amplitude of v(t) is given by sqrt(<v^2> - <v>^2)
    #the absolute amplitude is related to the RMS by sqrt(2.)
    #0.5*dv^2 =  c^2*A* <( 1 - m*sin(omega*t)^2 )> - vkms_mid^2
    #         =  c^2 * volt_mid - vkms_mid^2
    
    dvkms = np.sqrt( 2.*abs((c**2)*V - vkms_mid**2))
    windowWidths = 2*dvkms # dvkms above is the difference from the middle of the velocity bin to each side, the total width will be double
    vkms_lo = vkms_mid - dvkms
    vkms_hi = vkms_mid + dvkms
    
    vdf = totalFlux/(v_z*windowWidths)
    #vdf.putProperty(QDataSet.DEPEND_1,v_z)
    
    
    #norm = 100./total(ff.f*( (ff.v - shift(ff.v, 1))>0) > 0)
    norm = 62.415/1.31 # 1.31 is the cold/normal incidence sensor area in cm^2. 62.415 is 1/e times
                                    # unit conversions between
                                    # picoamp /(cm km/s)^2 and
                                    # coulomb /(km/s cm^3)
    
    vdf = vdf*norm
    
    return(vdf, v_z, windowWidths)


# You know how to get the total current and mv_lo, mv_hi from spc
totalFlux = 
mv_lo = 
mv_hi = 

(vdf, v_z, windowWidths) = ellipticalSPCVDF_MikeStyle(totalFlux, mv_lo, mv_hi)

import matplotlib.pyplot as plt
plt.plot(vdf)

