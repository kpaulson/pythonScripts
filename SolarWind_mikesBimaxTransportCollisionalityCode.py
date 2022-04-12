

# calculate the transport coefficients for coulomb collisions between two bi-maxwellian species according to 
# Hellinger & Travnicek Phys Plas 2009

# let species 1 and species 2 be ions with the parameters given
# this program calculates the rates dv/dt, dT_per/dt, and dT_par/dt for species 1 due to 
# collisions with species 2

# this calculation is performed in SI units and assumes SI inputs

# last revised: Mike Stevens 10/20/2011
# revision: K. Paulson 2022-04-08 - Adapted for Autoplot 
# revision: K. Paulson 2022-04-12 - Adapted for Python (works in 3.9) 




#########################################################################################
#########################################################################################



def bimax_transport(Q1, Q2, m1, m2, n1, n2, T1_per, T2_per, T1_par, T2_par, v1, v2, nu_12=0, decimal=2):

    import numpy as np
    from numpy import pi as PI

    #if not keyword_set(decimal) then decimal = 2
    if 'decimal' not in globals():
        decimal = 2
    
    # define necessary [SI] constants
    e0 = 8.85418781762E-12
    kbj= 1.3806503E-23
    qe = 1.60217646E-19
    
    # Just make sure the masses are not stored as integers (does dumb things with rounding)
    m1=float(m1)
    m2=float(m2)
    
    # calculate reduced quantities
    m12     = (m1/(m1+m2)) * m2  # Calculate the average reduced mass of combined distributions (Shouldn't this be weighted by the number densities?)
    w1_per  = np.sqrt(kbj * T1_per / m1)  # Calculate the perpendicular thermal speed of distribution 1
    w2_per  = np.sqrt(kbj * T2_per / m2)  # Calculate the perpendicular thermal speed of distribution 2
    w1_par  = np.sqrt(kbj * T1_par / m1)  # Calculate the parallel thermal speed of distribution 1
    w2_par  = np.sqrt(kbj * T2_par / m2)  # Calculate the parallel thermal speed of distribution 2
    v12_par = np.sqrt((w1_par**2 + w2_par**2)/2.)  # Calculate the average parallel thermal speed of both distributions
    v12_per = np.sqrt((w1_per**2 + w2_per**2)/2.)  # Calculate the average perpendicular thermal speed of both distributions
    
    # calculate the coulomb logarithm- 
    # assumption: electron temperature approximately equal to scalar ion temperature
    nplasma = n1+n2
    Tplasma = (n1*(T1_par + 2. * T1_per)/3. + n2*(T2_par + 2. * T2_per)/3.) / (n1 + n2)
    w1      = np.sqrt((2.* w1_per**2 + w1_par**2)/3.)
    w2      = np.sqrt((2.* w1_per**2 + w1_par**2)/3.)
    vab     = np.sqrt(w1**2 + w2**2 + (v2-v1)**2)
    ld      = np.sqrt( e0 * (kbj * Tplasma / qe) / (nplasma * qe) )
    ll      = Q1 * Q2 / (4 * PI * e0 * m2 * vab**2)
    C_log   = np.log(ld/ll)
    
    # calculate the collision frequency for species 1 on species 2
    nu_12 = C_log * (q1**2) * (q2**2) * n2 / ( 12. * (PI**(3./2.)) * (e0**2) * m1 * m12 * (v12_par**3) ) 
    
    # calculate the effective temperature anisotropy
    A12 = (v12_per**2) / (v12_par**2)
    
    # define the terms of the 2x hypergeometric functions
    x = 1. - A12
    y = A12 * ((v2-v1)**2)/(4. * v12_par**2)
    
    # double precision is not necessary here as x and y are order unity
    # and the HG function computation is not super high precision
    # We'll leave this here for now, but I think it is moot in Python (floats are double-precision)
    x = float(x)
    y = float(y) 
    
    # calculate the necessary hypergeometric functions
    F12_1_32_52 = np.exp(-((v2-v1)**2)/(4. * v12_par**2)) * Fxy_abc(x, y, 1., 1.5, 2.5, decimal = decimal)
    F12_1_12_52 = np.exp(-((v2-v1)**2)/(4. * v12_par**2)) * Fxy_abc(x, y, 1., 0.5, 2.5, decimal = decimal)
    F12_2_12_52 = np.exp(-((v2-v1)**2)/(4. * v12_par**2)) * Fxy_abc(x, y, 2., 0.5, 2.5, decimal = decimal)
    
    # unfortunately, we have a huge precision problem with the Fxy calculation
    fdiff = F12_2_12_52 - F12_1_12_52
    fsum = F12_2_12_52 + F12_1_12_52
    #if fdiff/fsum lt 0.1 then fdiff = 0.
    
    
    # calculate the transport rate for linear momentum
    nu_v1 = nu_12 * ((v2 - v1) / 2.) * F12_1_32_52
    
    # calculate the transport rate for parallel heat
    nu_T1_par = T1_par * nu_12 * (F12_1_12_52 * (m12/m2) * (T2_par/T1_par - 1.) # parallel-parallel exchange
                                  - 2. * fdiff             #  
                                  + F12_1_32_52 * ((v2-v1)**2)/(2.*v12_par**2) )  # parallel heating from flow
    
    # calculate the transport rate for perpendicular heat                              
    nu_T1_per = T1_per * (nu_12/A12) * (F12_2_12_52 * (m12/m2) * (T2_per/T1_per - 1.) # perp-perp exchange
                                        + fdiff)
    return(nu_v1, nu_T1_par, nu_T1_per)
    
#########################################################################################
#########################################################################################


# FUNCTION FOR CALCULATION OF SPECIFIC HYPERGEOMETRIC FUNCTIONS
def Fxy_abc( x, y, a, b, c, decimal):
    
    import numpy as np
    from numpy import outer as outerProduct
    import math

    # use the integral form to calculate the double hypergeometric function:
    #  2..    a, b,
    # F     ( c# b,    x,   y ) = X1 * integral(X2*X3*dt)
    #  1.1
    #
    
    if 'decimal' in globals():
        r = 10.**(decimal)
    else:
        r = 1e2
    
    # the integration variable is t. We integrate here numerically using "total"
    #t = findgen(int(r))/r  # IDL/Autoplot holdover. np.arange() below should be the analogue
    t = np.arange(int(r))/r 
    dt = 1./r
    
    # Again, I think this is moot in python. Are we really hurting for memory in other environments and need to specify 64-bit values?
#    if 'double' in locals():
#        a = float(a)
#        b = float(b)
#        c = float(c)
#        x = float(x)
#        y = float(y)
#        r = float(r)
#        dt = float(dt)
    
    #print( 't:',t )
    X1 = math.gamma(c)/(math.gamma(a)*math.gamma(c-a))
    #X2 = ((t##(1+0.*x))^(a-1.)) * ((1. - (t##(1+0.*x)))^(c-a-1.)) * (1. - t##x)^(0. - b)  # IDL version
    X2 = ((outerProduct(t,np.ones(1)))**(a-1.)) * ((1. - (outerProduct(t,np.ones(1))))**(c-a-1.)) * (1. - outerProduct(t,x))**(0. - b)
    #X3 = exp((t##y)/(1. - t##x))  # IDL version
    X3 = np.exp((outerProduct(t,y))/(1. - outerProduct(t,x)))
    
    return( (X1*np.sum(X2*X3*dt, axis=0))[0] )
    
    
#########################################################################################
#########################################################################################


# isotropization rate where X = tperp/tpar - 1
def iso_rate_coefficient( X, decimal):

    import numpy as np
    
    A12 = X+1
    x = 1. - A12
    y = 0*X
    
    # double precision is not necessary here as x and y are order unity
    # and the HG function computation is not super high precision
    x = float(x)
    y = float(y) 
    
    # calculate the necessary hypergeometric functions
    F12_1_12_52 = np.exp(-((v2-v1)**2)/(4. * v12_par**2)) * Fxy_abc(x, y, 1., 0.5, 2.5, decimal = decimal)
    F12_2_12_52 = np.exp(-((v2-v1)**2)/(4. * v12_par**2)) * Fxy_abc(x, y, 2., 0.5, 2.5, decimal = decimal)
    
    rate_coeff = Fdiff*(3.+2.*X)
    
    return(rate_coeff)

#########################################################################################
#########################################################################################




###
### pro testing
###

# find a-p collision rates for some typical SW conditions
e =  1.60217646E-19   # coulombs
mp = 1.67262158E-27   # kilograms
kbj = 1.3806503E-23    # Joules/Kelvin

q1 = e
q2 = 2.*e
m1 = mp
m2 = 4.*mp
n1 = 2E6              # m-3
n2 = 0.04*n1
T1_per = (m1/kbj) * (40. * 1E3)**2   # Kelvin (speed in m/s)
T1_par = (m1/kbj) * (50. * 1E3)**2
T2_per = 2.1*T1_per
T2_par = 2.8*T1_par
v1 = 0.
v2 = 10. * 1E3
nu_12 = 0.
decimal = 2
(nu_v1, nu_T1_par, nu_T1_per) = bimax_transport(q1, q2, m1, m2, n1, n2, T1_per, T2_per, T1_par, T2_par, v1, v2, nu_12=nu_12, decimal=decimal)

#stop

AU = 149598000.
u = 280. * 1E3
texp = AU/u
ncol = nu_T1_par * texp

print( '(nu_T1_par * texp):',(nu_T1_par * texp) )
print( '(nu_T1_per * texp):',nu_T1_per * texp )
print( '(nu_v1 * texp):',nu_v1 * texp )





# let's see how these numbers compare to Arnaud's
#nu=0.
#calculate_coll_freqs( kbj, n1, T1_per, m1, q1, m2, q2, n2, T2_per, (v2-v1), nu_slow, nu_perp, nu_para, nu_ener, nu0=nu0)
#print( nu_slow*texp )




