"""
Script to numerically integrate the equations (EOM) for a binary perturbed by a passing body on a hyperbolic orbit.
The EOM are averaged over the binary orbit, but not over the perturber's orbit. 
Valid to the second and third expansion orders in r_bin/r_per (i.e., quadrupole and octupole orders).

Adrian Hamers
June 2019
"""

import argparse
import numpy as np
import os

import pickle

from scipy.integrate import odeint

from wrapperflybyslibrary import flybyslibrary

def RHS_function(RHR_vec, theta, *ODE_args):
    """
    Singly-averaged (SA) equations of motion.
    """

    ### initialization ###
    eps_SA,eps_oct,e_per,args = ODE_args
    verbose = args.verbose

    ex = RHR_vec[0]
    ey = RHR_vec[1]
    ez = RHR_vec[2]

    jx = RHR_vec[3]
    jy = RHR_vec[4]
    jz = RHR_vec[5]
    
    dex_dtheta = dey_dtheta = dez_dtheta = djx_dtheta = djy_dtheta = djz_dtheta = 0.0
    
    
    if args.include_quadrupole_terms == True:
        dex_dtheta_quad,dey_dtheta_quad,dez_dtheta_quad,djx_dtheta_quad,djy_dtheta_quad,djz_dtheta_quad = dej_dtheta_quad(eps_SA,e_per,theta,ex,ey,ez,jx,jy,jz)
        
        dex_dtheta += dex_dtheta_quad
        dey_dtheta += dey_dtheta_quad
        dez_dtheta += dez_dtheta_quad
        djx_dtheta += djx_dtheta_quad
        djy_dtheta += djy_dtheta_quad
        djz_dtheta += djz_dtheta_quad

    if args.include_octupole_terms == True:
        dex_dtheta_oct,dey_dtheta_oct,dez_dtheta_oct,djx_dtheta_oct,djy_dtheta_oct,djz_dtheta_oct = dej_dtheta_oct(eps_SA,eps_oct,e_per,theta,ex,ey,ez,jx,jy,jz)
        
        dex_dtheta += dex_dtheta_oct
        dey_dtheta += dey_dtheta_oct
        dez_dtheta += dez_dtheta_oct
        djx_dtheta += djx_dtheta_oct
        djy_dtheta += djy_dtheta_oct
        djz_dtheta += djz_dtheta_oct


    if args.include_1PN_terms == True:
        e_vec = np.array([ex,ey,ez])
        j_vec = np.array([jx,jy,jz])
        j = np.linalg.norm(j_vec)

        rg = args.G*args.m/(args.c**2)
                
        eps_1PN = 3.0*(rg/args.a)*np.sqrt( (args.m/(args.m+args.M_per))*(args.Q/args.a)**3*(1.0+e_per)**3)
        de_dtheta = eps_1PN*np.cross(j_vec,e_vec)/(j**3*(1.0+e_per*np.cos(theta))**2)
	
        dex_dtheta += de_dtheta[0]
        dey_dtheta += de_dtheta[1]
        dez_dtheta += de_dtheta[2]
    

    RHR_vec_dot = [dex_dtheta,dey_dtheta,dez_dtheta,djx_dtheta,djy_dtheta,djz_dtheta]

    return RHR_vec_dot

def dej_dtheta_quad(eps_SA,e_per,theta,ex,ey,ez,jx,jy,jz):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    sin_2theta = np.sin(2.0*theta)

    dex_dtheta = (-3*eps_SA*(1 + e_per*cos_theta)*(3*ez*jy + ey*jz + (ez*jy - 5*ey*jz)*np.cos(2*theta) + (-(ez*jx) + 5*ex*jz)*sin_2theta))/4.
    dey_dtheta = (-3*eps_SA*(1 + e_per*cos_theta)*(-2*ez*jx + 2*ex*jz + (ez*jx - 5*ex*jz)*cos_theta**2 + (ez*jy - 5*ey*jz)*cos_theta*sin_theta))/2.
    dez_dtheta = (-3*eps_SA*(1 + e_per*cos_theta)*(-(ey*jx) + ex*jy + 2*(ey*jx + ex*jy)*np.cos(2*theta) + (-2*ex*jx + 2*ey*jy)*sin_2theta))/2.
    djx_dtheta = (3*eps_SA*(1 + e_per*cos_theta)*sin_theta*((-5*ex*ez + jx*jz)*cos_theta + (-5*ey*ez + jy*jz)*sin_theta))/2.
    djy_dtheta = (3*eps_SA*cos_theta*(1 + e_per*cos_theta)*((5*ex*ez - jx*jz)*cos_theta + (5*ey*ez - jy*jz)*sin_theta))/2.
    djz_dtheta = (3*eps_SA*(1 + e_per*cos_theta)*((-10*ex*ey + 2*jx*jy)*np.cos(2*theta) + (5*ex**2 - 5*ey**2 - jx**2 + jy**2)*sin_2theta))/4.

    return dex_dtheta,dey_dtheta,dez_dtheta,djx_dtheta,djy_dtheta,djz_dtheta

def dej_dtheta_oct(eps_SA,eps_oct,e_per,theta,ex,ey,ez,jx,jy,jz):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    sin_2theta = np.sin(2.0*theta)

    dex_dtheta = (15*eps_oct*eps_SA*(1 + e_per*cos_theta)**2*((1 - 8*(ex**2 + ey**2 + ez**2))*jz*sin_theta + 16*(ez*jy - ey*jz)*(ex*cos_theta + ey*sin_theta) + 35*jz*sin_theta*(ex*cos_theta + ey*sin_theta)**2 - 10*ez*sin_theta*(ex*cos_theta + ey*sin_theta)*(jx*cos_theta + jy*sin_theta) - 5*jz*sin_theta*(jx*cos_theta + jy*sin_theta)**2))/16.
    dey_dtheta = (15*eps_oct*eps_SA*(1 + e_per*cos_theta)**2*(-((1 - 8*(ex**2 + ey**2 + ez**2))*jz*cos_theta) + 16*(-(ez*jx) + ex*jz)*(ex*cos_theta + ey*sin_theta) - 35*jz*cos_theta*(ex*cos_theta + ey*sin_theta)**2 + 10*ez*cos_theta*(ex*cos_theta + ey*sin_theta)*(jx*cos_theta + jy*sin_theta) + 5*jz*cos_theta*(jx*cos_theta + jy*sin_theta)**2))/16.
    dez_dtheta = (15*eps_oct*eps_SA*(1 + e_per*cos_theta)**2*(16*(ey*jx - ex*jy)*(ex*cos_theta + ey*sin_theta) + 35*(ex*cos_theta + ey*sin_theta)**2*(jy*cos_theta - jx*sin_theta) - (1 - 8*(ex**2 + ey**2 + ez**2))*(-(jy*cos_theta) + jx*sin_theta) + 10*(-(ey*cos_theta) + ex*sin_theta)*(ex*cos_theta + ey*sin_theta)*(jx*cos_theta + jy*sin_theta) + 5*(-(jy*cos_theta) + jx*sin_theta)*(jx*cos_theta + jy*sin_theta)**2))/16.
    djx_dtheta = (15*eps_oct*eps_SA*(1 + e_per*cos_theta)**2*sin_theta*(ez - 8*ez*(ex**2 + ey**2 + ez**2) + 35*ez*(ex*cos_theta + ey*sin_theta)**2 - 10*jz*(ex*cos_theta + ey*sin_theta)*(jx*cos_theta + jy*sin_theta) - 5*ez*(jx*cos_theta + jy*sin_theta)**2))/16.
    djy_dtheta = (15*eps_oct*eps_SA*cos_theta*(1 + e_per*cos_theta)**2*(ez*(-1 + 8*(ex**2 + ey**2 + ez**2)) - 35*ez*(ex*cos_theta + ey*sin_theta)**2 + 10*jz*(ex*cos_theta + ey*sin_theta)*(jx*cos_theta + jy*sin_theta) + 5*ez*(jx*cos_theta + jy*sin_theta)**2))/16.
    djz_dtheta = (15*eps_oct*eps_SA*(1 + e_per*cos_theta)**2*((-1 + 8*(ex**2 + ey**2 + ez**2))*(-(ey*cos_theta) + ex*sin_theta) + 35*(ey*cos_theta - ex*sin_theta)*(ex*cos_theta + ey*sin_theta)**2 + 10*(ex*cos_theta + ey*sin_theta)*(-(jy*cos_theta) + jx*sin_theta)*(jx*cos_theta + jy*sin_theta) + 5*(-(ey*cos_theta) + ex*sin_theta)*(jx*cos_theta + jy*sin_theta)**2))/16.

    return dex_dtheta,dey_dtheta,dez_dtheta,djx_dtheta,djy_dtheta,djz_dtheta


def RHS_function_nbody(RHR_vec, theta, *ODE_args):
    """ Right-hand-side functions for 3-body integration.
    Not the most elegant way from a programming point of view, but it works.
    """
    
    ### initialization ###
    G,m1,m2,m3,args = ODE_args
    verbose = args.verbose


    ### initial state ###
    R1x = RHR_vec[0]
    R1y = RHR_vec[1]
    R1z = RHR_vec[2]
    V1x = RHR_vec[3]
    V1y = RHR_vec[4]
    V1z = RHR_vec[5]
    
    R2x = RHR_vec[6]
    R2y = RHR_vec[7]
    R2z = RHR_vec[8]
    V2x = RHR_vec[9]
    V2y = RHR_vec[10]
    V2z = RHR_vec[11]
    
    R3x = RHR_vec[12]
    R3y = RHR_vec[13]
    R3z = RHR_vec[14]
    V3x = RHR_vec[15]
    V3y = RHR_vec[16]
    V3z = RHR_vec[17]

    ### accelerations ###
    r12 = np.sqrt( (R1x-R2x)**2 + (R1y-R2y)**2 + (R1z-R2z)**2 )
    r13 = np.sqrt( (R1x-R3x)**2 + (R1y-R3y)**2 + (R1z-R3z)**2 )
    r23 = np.sqrt( (R2x-R3x)**2 + (R2y-R3y)**2 + (R2z-R3z)**2 )
    
    A1x = -G*m2*(R1x-R2x)/(r12**3) - G*m3*(R1x-R3x)/(r13**3)
    A2x = -G*m1*(R2x-R1x)/(r12**3) - G*m3*(R2x-R3x)/(r23**3)
    A3x = -G*m1*(R3x-R1x)/(r13**3) - G*m2*(R3x-R2x)/(r23**3)

    A1y = -G*m2*(R1y-R2y)/(r12**3) - G*m3*(R1y-R3y)/(r13**3)
    A2y = -G*m1*(R2y-R1y)/(r12**3) - G*m3*(R2y-R3y)/(r23**3)
    A3y = -G*m1*(R3y-R1y)/(r13**3) - G*m2*(R3y-R2y)/(r23**3)
    
    A1z = -G*m2*(R1z-R2z)/(r12**3) - G*m3*(R1z-R3z)/(r13**3)
    A2z = -G*m1*(R2z-R1z)/(r12**3) - G*m3*(R2z-R3z)/(r23**3)
    A3z = -G*m1*(R3z-R1z)/(r13**3) - G*m2*(R3z-R2z)/(r23**3)

    #print 'R1',R1x,R1y,R1z
    #print 'R2',R2x,R2y,R2z
    #print 'R3',R3x,R3y,R3z
    #print 'V3',V3x,V3y,V3z
    RHR_dot_vec = [V1x,V1y,V1z,A1x,A1y,A1z,V2x,V2y,V2z,A2x,A2y,A2z,V3x,V3y,V3z,A3x,A3y,A3z]

    return RHR_dot_vec

def orbital_elements_from_nbody(G,m,r,v):
    E = 0.5*np.dot(v,v) - G*m/np.linalg.norm(r)
    a = -G*m/(2.0*E)
    h = np.cross(r,v)
    e = (1.0/(G*m))*np.cross(v,h) - r/np.linalg.norm(r)
    e_norm = np.linalg.norm(e)
    i = np.arccos(h[2]/np.linalg.norm(h))
        
    return a,e_norm,i

def orbital_elements_to_orbital_vectors(e,i,omega,Omega):
    j = np.sqrt(1.0 - e**2)
    ex = e*(np.cos(omega)*np.cos(Omega) - np.cos(i)*np.sin(omega)*np.sin(Omega))
    ey = e*(np.cos(i)*np.cos(Omega)*np.sin(omega) + np.cos(omega)*np.sin(Omega))
    ez = e*np.sin(i)*np.sin(omega)
    jx = j*np.sin(i)*np.sin(Omega)
    jy = -j*np.cos(Omega)*np.sin(i)
    jz = j*np.cos(i)
    return ex,ey,ez,jx,jy,jz


def orbital_vectors_to_cartesian(G,m,a,theta_bin,ex,ey,ez,jx,jy,jz):
    e = np.sqrt(ex**2+ey**2+ez**2)
    e_hat_vec = np.array((ex,ey,ez))/e
    j_hat_vec = np.array((jx,jy,jz))/np.sqrt(1.0-e**2)
    q_hat_vec = np.cross(j_hat_vec,e_hat_vec)
    
    cos_theta_bin = np.cos(theta_bin)
    sin_theta_bin = np.sin(theta_bin)
    r_norm = a*(1.0-e**2)/(1.0 + e*cos_theta_bin)
    v_norm = np.sqrt(G*m/(a*(1.0-e**2)))
    r = np.zeros(3)
    v = np.zeros(3)

    for i in range(3):
        r[i] = r_norm*(e_hat_vec[i]*cos_theta_bin + q_hat_vec[i]*sin_theta_bin)
        v[i] = v_norm*(-sin_theta_bin*e_hat_vec[i] + (e+cos_theta_bin)*q_hat_vec[i])
    return r,v


def third_body_cartesian(G,m,M_per,Q,e_per,theta_0):
    a_per = Q/(e_per-1.0)

    M_tot = m+M_per

    n_per = np.sqrt(G*M_tot/(a_per**3))

    cos_true_anomaly = np.cos(theta_0)
    sin_true_anomaly = np.sin(theta_0)

    r_per = Q*(1.0 + e_per)/(1.0 + e_per*cos_true_anomaly);     
    r_dot_factor = np.sqrt(G*M_tot/(Q*(1.0 + e_per)))
   
    r_per_vec = np.zeros(3)
    r_dot_per_vec = np.zeros(3)
    e_per_hat_vec = np.array([1.0,0.0,0.0])
    q_per_hat_vec = np.array([0.0,1.0,0.0])
    j_per_hat_vec = np.array([0.0,0.0,1.0])
    
    for i in range(3):
        r_per_vec[i] = r_per*(cos_true_anomaly*e_per_hat_vec[i] + sin_true_anomaly*q_per_hat_vec[i])
        r_dot_per_vec[i] = r_dot_factor*( -sin_true_anomaly*e_per_hat_vec[i] + (e_per + cos_true_anomaly)*q_per_hat_vec[i])
    
    return r_per_vec,r_dot_per_vec

def compute_total_energy(G,m1,m2,m3,R1,V1,R2,V2,R3,V3):
    T = 0.5*m1*np.dot(V1,V1) + 0.5*m2*np.dot(V2,V2) + 0.5*m3*np.dot(V3,V3)
    V = -G*m1*m2/np.linalg.norm(R1-R2) - G*m1*m3/np.linalg.norm(R1-R3) - G*m2*m3/np.linalg.norm(R2-R3)
    return T+V
    
def integrate(args):
    """
    Integrate singly-averaged (SA) equations of motion.
    """
    
    ### initial conditions ###   
    m = args.m
    m1 = args.m1
    m2 = args.m2
    M_per = args.M_per
    e_per = args.e_per
    Q = args.Q
    
    a = args.a
    e = args.e

    eps_SA = compute_eps_SA(m,M_per,a/Q,e_per)
    eps_oct = compute_eps_oct(m1,m2,m,a/Q,e_per)

    #print 'eps_SA',eps_SA
    i = args.i
    omega = args.omega
    Omega = args.Omega
    
    L = args.fraction_theta_0*np.arccos(-1.0/e_per)

    if 1==0:
        if e_per==1.0:
            e_per += 1.0e-10
        a_per = Q/(e_per-1.0)
        M_tot = m+M_per
        n_per = np.sqrt(args.G*M_tot/(a_per**3))

        a = args.fraction_theta_0*L
        Delta_t = (1.0/n_per)*( -4*np.arctanh(((-1 + e_per)*np.tan(a/2.))/np.sqrt(-1 + e_per**2)) + (2*e_per*np.sqrt(-1 + e_per**2)*np.sin(a))/(1 + e_per*np.cos(a)) )
        n_bin = np.sqrt(args.G*m/(a**3))
        rg = args.G*m/(args.c**2)
        omega_dot_1PN = 3.0*n_bin*(rg/a)*1.0/(1.0-e**2)
        Delta_omega = omega_dot_1PN*Delta_t/2.0
        print( 'Delta_omega',Delta_omega,'Delta_t',Delta_t)
        omega += omega_dot_1PN
    
    ex,ey,ez,jx,jy,jz = orbital_elements_to_orbital_vectors(e,i,omega,Omega)
        
    N_steps = args.N_steps
    #theta_0 = np.arccos(-1.0/e_per)
    thetas = np.linspace(-L, L, N_steps)

    ODE_args = (eps_SA,eps_oct,e_per,args)

    RHR_vec = [ex,ey,ez,jx,jy,jz]
        
    if args.verbose==True:
        print( 'eps_SA',eps_SA)
        print( 'RHR_vec',RHR_vec)
    
    ### numerical solution ###
    sol = odeint(RHS_function, RHR_vec, thetas, args=ODE_args,mxstep=args.mxstep,rtol=1.0e-15,atol=1e-12)
    
    ex_sol = np.array(sol[:,0])
    ey_sol = np.array(sol[:,1])
    ez_sol = np.array(sol[:,2])
    jx_sol = np.array(sol[:,3])
    jy_sol = np.array(sol[:,4])
    jz_sol = np.array(sol[:,5])

    e_sol = [np.sqrt(ex_sol[i]**2 + ey_sol[i]**2 + ez_sol[i]**2) for i in range(len(thetas))]
    j_sol = [np.sqrt(jx_sol[i]**2 + jy_sol[i]**2 + jz_sol[i]**2) for i in range(len(thetas))]
    i_sol = [np.arccos(jz_sol[i]/j_sol[i]) for i in range(len(thetas))]
    
    if 1==0:
        from matplotlib import pyplot
        fig=pyplot.figure()
        plot=fig.add_subplot(1,1,1)
        plot.scatter(ex_sol,ey_sol)
        pyplot.show()
    
    Delta_e = e_sol[-1] - e_sol[0]
    Delta_i = i_sol[-1] - i_sol[0]

    nbody_a_sol,nbody_e_sol,nbody_i_sol,nbody_Delta_a,nbody_Delta_e,nbody_Delta_i,nbody_energy_errors_sol = None,None,None,None,None,None,None
    if args.do_nbody==True:
        nbody_a_sol,nbody_e_sol,nbody_i_sol,nbody_Delta_a,nbody_Delta_e,nbody_Delta_i,nbody_energy_errors_sol = integrate_nbody(args)
        print( 'Q/a',args.Q/args.a)
        print( 'SA Delta e',Delta_e,'Delta i',Delta_i)
        print( 'nbody Delta e',nbody_Delta_e,'Delta i',nbody_Delta_i,'nbody energy error',nbody_energy_errors_sol[-1])

    data = {'thetas': thetas,'ex_sol':ex_sol,'ey_sol':ey_sol,'ez_sol':ez_sol, \
        'jx_sol':jx_sol,'jy_sol':jy_sol,'jz_sol':jz_sol,'e_sol':e_sol,'j_sol':j_sol,'i_sol':i_sol, \
        'Delta_e':Delta_e,'Delta_i':Delta_i, \
        'do_nbody':args.do_nbody,'nbody_a_sol':nbody_a_sol,'nbody_e_sol':nbody_e_sol,'nbody_i_sol':nbody_i_sol, \
        'nbody_Delta_a':nbody_Delta_a,'nbody_Delta_e':nbody_Delta_e,'nbody_Delta_i':nbody_Delta_i,'nbody_energy_errors_sol':nbody_energy_errors_sol}
    
    filename = args.data_filename

    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data

def integrate_nbody(args):

    """
    Integrate 3-body equations of motion.
    """

    ### initial conditions ###   
    G = args.G
    theta_bin = args.theta_bin
    m = args.m
    m1 = args.m1
    m2 = args.m2
    M_per = args.M_per
    epsilon=1.0e-5
    e_per = args.e_per+epsilon
    Q = args.Q
    
    a = args.a
    e = args.e

    eps_SA = compute_eps_SA(m,M_per,a/Q,e_per)
    eps_oct = compute_eps_oct(m1,m2,m,a/Q,e_per)
    R = np.sqrt( (1.0+M_per/m)*(a/Q)**3*(1.0+e_per) ) ### `secular' ratio ###

    i = args.i
    omega = args.omega
    Omega = args.Omega
    
    ex,ey,ez,jx,jy,jz = orbital_elements_to_orbital_vectors(e,i,omega,Omega)
    r,v = orbital_vectors_to_cartesian(G,m,a,theta_bin,ex,ey,ez,jx,jy,jz)
    

    R_cm, V_cm = np.zeros(3), np.zeros(3)
    theta_0 = -args.fraction_theta_0*np.arccos(-1.0/e_per)
    R3_rel,V3_rel = third_body_cartesian(G,m,M_per,Q,e_per,theta_0)

    R_cm = (M_per/(m+M_per))*R3_rel
    V_cm = (M_per/(m+M_per))*V3_rel

    R1 = R_cm + (m2/m)*r
    R2 = R_cm - (m1/m)*r
    V1 = V_cm + (m2/m)*v
    V2 = V_cm - (m1/m)*v

    R3 = -(m/(m+M_per))*R3_rel
    V3 = -(m/(m+M_per))*V3_rel
    #print 'R1',R1,'R2',R2,'R3',R3
    #print 'V1',V1,'V2',V2,'V3',V3

    N_steps = args.N_steps
    
    a_per = Q/(e_per-1.0)
    M_tot = m+M_per
    

    n_per = np.sqrt(G*M_tot/(a_per**3))
    
    a = -theta_0
    tend = (1.0/n_per)*( -4*np.arctanh(((-1 + e_per)*np.tan(a/2.))/np.sqrt(-1 + e_per**2)) + (2*e_per*np.sqrt(-1 + e_per**2)*np.sin(a))/(1 + e_per*np.cos(a)) )
    
    #print 'tend',tend,'P_bin',2.0*np.pi*np.sqrt(a**3/(G*M)),(1.0/n_per)
    
    times = np.linspace(0.0,tend,N_steps)

    ODE_args = (G,m1,m2,M_per,args)

    RHR_vec = [R1[0],R1[1],R1[2],V1[0],V1[1],V1[2],R2[0],R2[1],R2[2],V2[0],V2[1],V2[2],R3[0],R3[1],R3[2],V3[0],V3[1],V3[2]]

    if args.verbose==True:
        print( 'eps_SA',eps_SA)
        print( 'RHR_vec',RHR_vec)
    
    ### numerical solution ###
    sol = odeint(RHS_function_nbody, RHR_vec, times, args=ODE_args,mxstep=args.mxstep,rtol=1.0e-15)
    
    R1_sol = np.transpose(np.array([sol[:,0],sol[:,1],sol[:,2]]))
    V1_sol = np.transpose(np.array([sol[:,3],sol[:,4],sol[:,5]]))
    R2_sol = np.transpose(np.array([sol[:,6],sol[:,7],sol[:,8]]))
    V2_sol = np.transpose(np.array([sol[:,9],sol[:,10],sol[:,11]]))
    R3_sol = np.transpose(np.array([sol[:,12],sol[:,13],sol[:,14]]))
    V3_sol = np.transpose(np.array([sol[:,15],sol[:,16],sol[:,17]]))
    
    r_sol = R1_sol - R2_sol
    v_sol = V1_sol - V2_sol

    a_sol = []
    e_sol = []
    energy_errors_sol = []
    i_sol = []
    
    for index_t,t in enumerate(times):
        a,e,i = orbital_elements_from_nbody(G,m1+m2,r_sol[index_t],v_sol[index_t])
        a_sol.append(a)
        e_sol.append(e)
        i_sol.append(i)
    
        energy = compute_total_energy(G,m1,m2,M_per,R1_sol[index_t],V1_sol[index_t],R2_sol[index_t],V2_sol[index_t],R3_sol[index_t],V3_sol[index_t])

        if index_t==0:
            initial_energy = energy
        energy_errors_sol.append( np.fabs((initial_energy-energy)/initial_energy) )
        
    Delta_a = a_sol[-1] - a_sol[0]
    Delta_e = e_sol[-1] - e_sol[0]
    Delta_i = i_sol[-1] - i_sol[0]
    
    if 1==0: ### turn on to see detailed 3-body results ###
        fig=pyplot.figure()
        plot1=fig.add_subplot(3,1,1)
        plot2=fig.add_subplot(3,1,2)
        plot3=fig.add_subplot(3,1,3,yscale="log")    
        plot1.plot(times,a_sol)
        plot2.plot(times,e_sol)
        plot3.plot(times,energy_errors_sol)
        pyplot.show()

    return a_sol,e_sol,i_sol,Delta_a,Delta_e,Delta_i,energy_errors_sol

    
def fourier_series_ej(lmax,theta,eps_SA,e_per,ex0,ey0,ez0,jx0,jy0,jz0):
    
    ArcCos = np.arccos
    Sqrt = np.sqrt
    eper = e_per
    Pi = np.pi
    Cos = np.cos
    Sin = np.sin

    f1 = f(1.0,e_per)
    f2 = f(2.0,e_per)
    f3 = f(3.0,e_per)

    
    if lmax==1:
        ex = ex0 + eps_SA*(((theta + ArcCos(-(1/eper)))*(Sqrt(1 - eper**(-2))*(ez0*(jy0 - 10*eper**2*jy0) + (-5 + 2*eper**2)*ey0*jz0) - 3*eper*(3*ez0*jy0 + ey0*jz0)*ArcCos(-1.0/eper)))/(4.*eper*ArcCos(-(1/eper))) - (3*Sqrt(1 - eper**(-2))*(ez0*jx0 - 5*ex0*jz0)*ArcCos(-(1/eper))**3*((5 - 2*eper**2)*Pi**2 + (-5 + 8*eper**2)*ArcCos(-(1/eper))**2))/(eper*f1) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*((ez0*jx0 - 5*ex0*jz0)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper))) + (-(((1 + eper**2)*ez0*jy0 + (-5 + 3*eper**2)*ey0*jz0)*Pi**4) + ((-5 + 19*eper**2)*ez0*jy0 + (25 + 9*eper**2)*ey0*jz0)*Pi**2*ArcCos(-(1/eper))**2 - 6*((-1 + 10*eper**2)*ez0*jy0 + (5 - 2*eper**2)*ey0*jz0)*ArcCos(-(1/eper))**4)*Sin((Pi*theta)/ArcCos(-(1/eper)))))/(eper*Pi*f1))
        ey = ey0 + eps_SA*(((theta + ArcCos(-(1/eper)))*(Sqrt(1 - eper**(-2))*(ez0*(jx0 + 8*eper**2*jx0) + (-5 + 8*eper**2)*ex0*jz0) + 3*eper*(3*ez0*jx0 + ex0*jz0)*ArcCos(-1.0/eper)))/(4.*eper*ArcCos(-(1/eper))) + (3*Sqrt(1 - eper**(-2))*(ez0*jy0 - 5*ey0*jz0)*ArcCos(-(1/eper))**3*((5 - 2*eper**2)*Pi**2 + (-5 + 8*eper**2)*ArcCos(-(1/eper))**2))/(eper*f1) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*(-((ez0*jy0 - 5*ey0*jz0)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper)))) - ((ez0*(jx0 - 2*eper**2*jx0) + (-5 + 2*eper**2)*ex0*jz0)*Pi**4 + (5*ez0*(jx0 + 4*eper**2*jx0) + (-25 + 4*eper**2)*ex0*jz0)*Pi**2*ArcCos(-(1/eper))**2 - 6*(ez0*(jx0 + 8*eper**2*jx0) + (-5 + 8*eper**2)*ex0*jz0)*ArcCos(-(1/eper))**4)*Sin((Pi*theta)/ArcCos(-(1/eper)))))/(eper*Pi*f1))
        ez = ez0 + eps_SA*(((3*(ey0*jx0 - ex0*jy0))/2. + (Sqrt(1 - eper**(-2))*((2 + eper**2)*ey0*jx0 + (2 - 5*eper**2)*ex0*jy0))/(2.*eper*ArcCos(-(1/eper))))*(theta + ArcCos(-(1/eper))) - (12*Sqrt(1 - eper**(-2))*(ex0*jx0 - ey0*jy0)*ArcCos(-(1/eper))**3*((5 - 2*eper**2)*Pi**2 + (-5 + 8*eper**2)*ArcCos(-(1/eper))**2))/(eper*f1) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*(4*(ex0*jx0 - ey0*jy0)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper))) - (((4 - 3*eper**2)*ey0*jx0 - (-4 + eper**2)*ex0*jy0)*Pi**4 + (5*(4 + 3*eper**2)*ey0*jx0 + (20 - 11*eper**2)*ex0*jy0)*Pi**2*ArcCos(-(1/eper))**2 - 12*((2 + eper**2)*ey0*jx0 + (2 - 5*eper**2)*ex0*jy0)*ArcCos(-(1/eper))**4)*Sin((Pi*theta)/ArcCos(-(1/eper)))))/(eper*Pi*f1))
        jx = jx0 + eps_SA*(-((5*ey0*ez0 - jy0*jz0)*(theta + ArcCos(-(1/eper)))*(Sqrt(1 - eper**(-2))*(1 + 2*eper**2) + 3*eper*ArcCos(-(1/eper))))/(4.*eper*ArcCos(-(1/eper))) + (3*Sqrt(1 - eper**(-2))*(5*ex0*ez0 - jx0*jz0)*ArcCos(-(1/eper))**3*((5 - 2*eper**2)*Pi**2 + (-5 + 8*eper**2)*ArcCos(-(1/eper))**2))/(eper*f1) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*(-((5*ex0*ez0 - jx0*jz0)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper)))) - (5*ey0*ez0 - jy0*jz0)*((-1 + eper**2)*Pi**4 - (5 + 7*eper**2)*Pi**2*ArcCos(-(1/eper))**2 + 6*(1 + 2*eper**2)*ArcCos(-(1/eper))**4)*Sin((Pi*theta)/ArcCos(-(1/eper)))))/(eper*Pi*f1))
        jy = jy0 + eps_SA*(((5*ex0*ez0 - jx0*jz0)*(theta + ArcCos(-(1/eper)))*(Sqrt(1 - eper**(-2))*(-1 + 4*eper**2) + 3*eper*ArcCos(-(1/eper))))/(4.*eper*ArcCos(-(1/eper))) - (3*Sqrt(1 - eper**(-2))*(5*ey0*ez0 - jy0*jz0)*ArcCos(-(1/eper))**3*((5 - 2*eper**2)*Pi**2 + (-5 + 8*eper**2)*ArcCos(-(1/eper))**2))/(eper*f1) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*((5*ey0*ez0 - jy0*jz0)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper))) - (5*ex0*ez0 - jx0*jz0)*(-Pi**4 + (-5 + 6*eper**2)*Pi**2*ArcCos(-(1/eper))**2 + (6 - 24*eper**2)*ArcCos(-(1/eper))**4)*Sin((Pi*theta)/ArcCos(-(1/eper)))))/(eper*Pi*f1))
        jz = jz0 + eps_SA*(-(Sqrt(1 - eper**(-2))*(-1 + eper**2)*(5*ex0*ey0 - jx0*jy0)*(theta + ArcCos(-(1/eper))))/(2.*eper*ArcCos(-(1/eper))) - (3*Sqrt(1 - eper**(-2))*(5*ex0**2 - 5*ey0**2 - jx0**2 + jy0**2)*ArcCos(-(1/eper))**3*((5 - 2*eper**2)*Pi**2 + (-5 + 8*eper**2)*ArcCos(-(1/eper))**2))/(eper*f1) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*((5*ex0**2 - 5*ey0**2 - jx0**2 + jy0**2)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper))) + (5*ex0*ey0 - jx0*jy0)*((-2 + eper**2)*Pi**4 - (10 + eper**2)*Pi**2*ArcCos(-(1/eper))**2 - 12*(-1 + eper**2)*ArcCos(-(1/eper))**4)*Sin((Pi*theta)/ArcCos(-(1/eper)))))/(eper*Pi*f1))
    elif lmax==2:
        ex = ex0 + eps_SA*(((theta + ArcCos(-(1/eper)))*(Sqrt(1 - eper**(-2))*(ez0*(jy0 - 10*eper**2*jy0) + (-5 + 2*eper**2)*ey0*jz0) - 3*eper*(3*ez0*jy0 + ey0*jz0)*ArcCos(-1.0/eper)))/(4.*eper*ArcCos(-(1/eper))) - (3*Sqrt(1 - eper**(-2))*(ez0*jx0 - 5*ex0*jz0)*ArcCos(-(1/eper))**3*((-5 + 8*eper**2)*ArcCos(-(1/eper))**2*(f1 + f2) - (-5 + 2*eper**2)*Pi**2*(4*f1 + f2)))/(eper*f1*f2) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*(-((ez0*jx0 - 5*ex0*jz0)*Pi*ArcCos(-(1/eper))*(4*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((2*Pi*theta)/ArcCos(-(1/eper)))*f1) + (ez0*jx0 - 5*ex0*jz0)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper)))*f2 - (((1 + eper**2)*ez0*jy0 + (-5 + 3*eper**2)*ey0*jz0)*Pi**4 - ((-5 + 19*eper**2)*ez0*jy0 + (25 + 9*eper**2)*ey0*jz0)*Pi**2*ArcCos(-(1/eper))**2 + 6*((-1 + 10*eper**2)*ez0*jy0 + (5 - 2*eper**2)*ey0*jz0)*ArcCos(-(1/eper))**4)*f2*Sin((Pi*theta)/ArcCos(-(1/eper))) + (8*((1 + eper**2)*ez0*jy0 + (-5 + 3*eper**2)*ey0*jz0)*Pi**4 - 2*((-5 + 19*eper**2)*ez0*jy0 + (25 + 9*eper**2)*ey0*jz0)*Pi**2*ArcCos(-(1/eper))**2 + 3*((-1 + 10*eper**2)*ez0*jy0 + (5 - 2*eper**2)*ey0*jz0)*ArcCos(-(1/eper))**4)*f1*Sin((2*Pi*theta)/ArcCos(-(1/eper)))))/(eper*Pi*f1*f2))
        ey = ey0 + eps_SA*(((theta + ArcCos(-(1/eper)))*(Sqrt(1 - eper**(-2))*(ez0*(jx0 + 8*eper**2*jx0) + (-5 + 8*eper**2)*ex0*jz0) + 3*eper*(3*ez0*jx0 + ex0*jz0)*ArcCos(-1.0/eper)))/(4.*eper*ArcCos(-(1/eper))) + (3*Sqrt(1 - eper**(-2))*(ez0*jy0 - 5*ey0*jz0)*ArcCos(-(1/eper))**3*((-5 + 8*eper**2)*ArcCos(-(1/eper))**2*(f1 + f2) - (-5 + 2*eper**2)*Pi**2*(4*f1 + f2)))/(eper*f1*f2) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*((ez0*jy0 - 5*ey0*jz0)*Pi*ArcCos(-(1/eper))*(4*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((2*Pi*theta)/ArcCos(-(1/eper)))*f1 - (ez0*jy0 - 5*ey0*jz0)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper)))*f2 + (((-1 + 2*eper**2)*ez0*jx0 + (5 - 2*eper**2)*ex0*jz0)*Pi**4 - (5*ez0*(jx0 + 4*eper**2*jx0) + (-25 + 4*eper**2)*ex0*jz0)*Pi**2*ArcCos(-(1/eper))**2 + 6*(ez0*(jx0 + 8*eper**2*jx0) + (-5 + 8*eper**2)*ex0*jz0)*ArcCos(-(1/eper))**4)*f2*Sin((Pi*theta)/ArcCos(-(1/eper))) - (8*((-1 + 2*eper**2)*ez0*jx0 + (5 - 2*eper**2)*ex0*jz0)*Pi**4 - 2*(5*ez0*(jx0 + 4*eper**2*jx0) + (-25 + 4*eper**2)*ex0*jz0)*Pi**2*ArcCos(-(1/eper))**2 + 3*(ez0*(jx0 + 8*eper**2*jx0) + (-5 + 8*eper**2)*ex0*jz0)*ArcCos(-(1/eper))**4)*f1*Sin((2*Pi*theta)/ArcCos(-(1/eper)))))/(eper*Pi*f1*f2))
        ez = ez0 + eps_SA*(((3*(ey0*jx0 - ex0*jy0))/2. + (Sqrt(1 - eper**(-2))*((2 + eper**2)*ey0*jx0 + (2 - 5*eper**2)*ex0*jy0))/(2.*eper*ArcCos(-(1/eper))))*(theta + ArcCos(-(1/eper))) - (12*Sqrt(1 - eper**(-2))*(ex0*jx0 - ey0*jy0)*ArcCos(-(1/eper))**3*((-5 + 8*eper**2)*ArcCos(-(1/eper))**2*(f1 + f2) - (-5 + 2*eper**2)*Pi**2*(4*f1 + f2)))/(eper*f1*f2) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*(-4*(ex0*jx0 - ey0*jy0)*Pi*ArcCos(-(1/eper))*(4*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((2*Pi*theta)/ArcCos(-(1/eper)))*f1 + 4*(ex0*jx0 - ey0*jy0)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper)))*f2 + (((-4 + 3*eper**2)*ey0*jx0 + (-4 + eper**2)*ex0*jy0)*Pi**4 - (5*(4 + 3*eper**2)*ey0*jx0 + (20 - 11*eper**2)*ex0*jy0)*Pi**2*ArcCos(-(1/eper))**2 + 12*((2 + eper**2)*ey0*jx0 + (2 - 5*eper**2)*ex0*jy0)*ArcCos(-(1/eper))**4)*f2*Sin((Pi*theta)/ArcCos(-(1/eper))) - 2*(4*((-4 + 3*eper**2)*ey0*jx0 + (-4 + eper**2)*ex0*jy0)*Pi**4 - (5*(4 + 3*eper**2)*ey0*jx0 + (20 - 11*eper**2)*ex0*jy0)*Pi**2*ArcCos(-(1/eper))**2 + 3*((2 + eper**2)*ey0*jx0 + (2 - 5*eper**2)*ex0*jy0)*ArcCos(-(1/eper))**4)*f1*Sin((2*Pi*theta)/ArcCos(-(1/eper)))))/(eper*Pi*f1*f2))
        jx = jx0 + eps_SA*(-((5*ey0*ez0 - jy0*jz0)*(theta + ArcCos(-(1/eper)))*(Sqrt(1 - eper**(-2))*(1 + 2*eper**2) + 3*eper*ArcCos(-(1/eper))))/(4.*eper*ArcCos(-(1/eper))) + (3*Sqrt(1 - eper**(-2))*(5*ex0*ez0 - jx0*jz0)*ArcCos(-(1/eper))**3*((-5 + 8*eper**2)*ArcCos(-(1/eper))**2*(f1 + f2) - (-5 + 2*eper**2)*Pi**2*(4*f1 + f2)))/(eper*f1*f2) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*((5*ex0*ez0 - jx0*jz0)*Pi*ArcCos(-(1/eper))*(4*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((2*Pi*theta)/ArcCos(-(1/eper)))*f1 - (5*ex0*ez0 - jx0*jz0)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper)))*f2 - (5*ey0*ez0 - jy0*jz0)*((-1 + eper**2)*Pi**4 - (5 + 7*eper**2)*Pi**2*ArcCos(-(1/eper))**2 + 6*(1 + 2*eper**2)*ArcCos(-(1/eper))**4)*f2*Sin((Pi*theta)/ArcCos(-(1/eper))) + (5*ey0*ez0 - jy0*jz0)*(8*(-1 + eper**2)*Pi**4 - 2*(5 + 7*eper**2)*Pi**2*ArcCos(-(1/eper))**2 + (3 + 6*eper**2)*ArcCos(-(1/eper))**4)*f1*Sin((2*Pi*theta)/ArcCos(-(1/eper)))))/(eper*Pi*f1*f2))
        jy = jy0 + eps_SA*(((5*ex0*ez0 - jx0*jz0)*(theta + ArcCos(-(1/eper)))*(Sqrt(1 - eper**(-2))*(-1 + 4*eper**2) + 3*eper*ArcCos(-(1/eper))))/(4.*eper*ArcCos(-(1/eper))) - (3*Sqrt(1 - eper**(-2))*(5*ey0*ez0 - jy0*jz0)*ArcCos(-(1/eper))**3*((-5 + 8*eper**2)*ArcCos(-(1/eper))**2*(f1 + f2) - (-5 + 2*eper**2)*Pi**2*(4*f1 + f2)))/(eper*f1*f2) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*(-((5*ey0*ez0 - jy0*jz0)*Pi*ArcCos(-(1/eper))*(4*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((2*Pi*theta)/ArcCos(-(1/eper)))*f1) + (5*ey0*ez0 - jy0*jz0)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper)))*f2 + (5*ex0*ez0 - jx0*jz0)*(Pi**4 + (5 - 6*eper**2)*Pi**2*ArcCos(-(1/eper))**2 + 6*(-1 + 4*eper**2)*ArcCos(-(1/eper))**4)*f2*Sin((Pi*theta)/ArcCos(-(1/eper))) - (5*ex0*ez0 - jx0*jz0)*(8*Pi**4 - 2*(-5 + 6*eper**2)*Pi**2*ArcCos(-(1/eper))**2 + 3*(-1 + 4*eper**2)*ArcCos(-(1/eper))**4)*f1*Sin((2*Pi*theta)/ArcCos(-(1/eper)))))/(eper*Pi*f1*f2))
        jz = jz0 + eps_SA*(-(Sqrt(1 - eper**(-2))*(-1 + eper**2)*(5*ex0*ey0 - jx0*jy0)*(theta + ArcCos(-(1/eper))))/(2.*eper*ArcCos(-(1/eper))) - (3*Sqrt(1 - eper**(-2))*(5*ex0**2 - 5*ey0**2 - jx0**2 + jy0**2)*ArcCos(-(1/eper))**3*((-5 + 8*eper**2)*ArcCos(-(1/eper))**2*(f1 + f2) - (-5 + 2*eper**2)*Pi**2*(4*f1 + f2)))/(eper*f1*f2) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*(-((5*ex0**2 - 5*ey0**2 - jx0**2 + jy0**2)*Pi*ArcCos(-(1/eper))*(4*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((2*Pi*theta)/ArcCos(-(1/eper)))*f1) + (5*ex0**2 - 5*ey0**2 - jx0**2 + jy0**2)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper)))*f2 + (5*ex0*ey0 - jx0*jy0)*((-2 + eper**2)*Pi**4 - (10 + eper**2)*Pi**2*ArcCos(-(1/eper))**2 - 12*(-1 + eper**2)*ArcCos(-(1/eper))**4)*f2*Sin((Pi*theta)/ArcCos(-(1/eper))) - 2*(5*ex0*ey0 - jx0*jy0)*(4*(-2 + eper**2)*Pi**4 - (10 + eper**2)*Pi**2*ArcCos(-(1/eper))**2 - 3*(-1 + eper**2)*ArcCos(-(1/eper))**4)*f1*Sin((2*Pi*theta)/ArcCos(-(1/eper)))))/(eper*Pi*f1*f2))
    elif lmax==3:
        ex = ex0 + eps_SA*(((theta + ArcCos(-(1/eper)))*(Sqrt(1 - eper**(-2))*(ez0*(jy0 - 10*eper**2*jy0) + (-5 + 2*eper**2)*ey0*jz0) - 3*eper*(3*ez0*jy0 + ey0*jz0)*ArcCos(-1.0/eper)))/(4.*eper*ArcCos(-(1/eper))) - (3*Sqrt(1 - eper**(-2))*(ez0*jx0 - 5*ex0*jz0)*ArcCos(-(1/eper))**3*((-5 + 8*eper**2)*ArcCos(-(1/eper))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*eper**2)*Pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(eper*f1*f2*f3) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*(((ez0*jx0 - 5*ex0*jz0)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper))))/f1 - ((ez0*jx0 - 5*ex0*jz0)*Pi*ArcCos(-(1/eper))*(4*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((2*Pi*theta)/ArcCos(-(1/eper))))/f2 + ((ez0*jx0 - 5*ex0*jz0)*Pi*ArcCos(-(1/eper))*(9*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((3*Pi*theta)/ArcCos(-(1/eper))))/f3 - ((((1 + eper**2)*ez0*jy0 + (-5 + 3*eper**2)*ey0*jz0)*Pi**4 - ((-5 + 19*eper**2)*ez0*jy0 + (25 + 9*eper**2)*ey0*jz0)*Pi**2*ArcCos(-(1/eper))**2 + 6*((-1 + 10*eper**2)*ez0*jy0 + (5 - 2*eper**2)*ey0*jz0)*ArcCos(-(1/eper))**4)*Sin((Pi*theta)/ArcCos(-(1/eper))))/f1 + ((8*((1 + eper**2)*ez0*jy0 + (-5 + 3*eper**2)*ey0*jz0)*Pi**4 - 2*((-5 + 19*eper**2)*ez0*jy0 + (25 + 9*eper**2)*ey0*jz0)*Pi**2*ArcCos(-(1/eper))**2 + 3*((-1 + 10*eper**2)*ez0*jy0 + (5 - 2*eper**2)*ey0*jz0)*ArcCos(-(1/eper))**4)*Sin((2*Pi*theta)/ArcCos(-(1/eper))))/f2 - ((27*((1 + eper**2)*ez0*jy0 + (-5 + 3*eper**2)*ey0*jz0)*Pi**4 - 3*((-5 + 19*eper**2)*ez0*jy0 + (25 + 9*eper**2)*ey0*jz0)*Pi**2*ArcCos(-(1/eper))**2 + 2*((-1 + 10*eper**2)*ez0*jy0 + (5 - 2*eper**2)*ey0*jz0)*ArcCos(-(1/eper))**4)*Sin((3*Pi*theta)/ArcCos(-(1/eper))))/f3))/(eper*Pi))
        ey = ey0 + eps_SA*(((theta + ArcCos(-(1/eper)))*(Sqrt(1 - eper**(-2))*(ez0*(jx0 + 8*eper**2*jx0) + (-5 + 8*eper**2)*ex0*jz0) + 3*eper*(3*ez0*jx0 + ex0*jz0)*ArcCos(-1.0/eper)))/(4.*eper*ArcCos(-(1/eper))) + (3*Sqrt(1 - eper**(-2))*(ez0*jy0 - 5*ey0*jz0)*ArcCos(-(1/eper))**3*((-5 + 8*eper**2)*ArcCos(-(1/eper))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*eper**2)*Pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(eper*f1*f2*f3) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*(-(((ez0*jy0 - 5*ey0*jz0)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper))))/f1) + ((ez0*jy0 - 5*ey0*jz0)*Pi*ArcCos(-(1/eper))*(4*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((2*Pi*theta)/ArcCos(-(1/eper))))/f2 - ((ez0*jy0 - 5*ey0*jz0)*Pi*ArcCos(-(1/eper))*(9*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((3*Pi*theta)/ArcCos(-(1/eper))))/f3 + ((((-1 + 2*eper**2)*ez0*jx0 + (5 - 2*eper**2)*ex0*jz0)*Pi**4 - (5*ez0*(jx0 + 4*eper**2*jx0) + (-25 + 4*eper**2)*ex0*jz0)*Pi**2*ArcCos(-(1/eper))**2 + 6*(ez0*(jx0 + 8*eper**2*jx0) + (-5 + 8*eper**2)*ex0*jz0)*ArcCos(-(1/eper))**4)*Sin((Pi*theta)/ArcCos(-(1/eper))))/f1 - ((-8*(ez0*(jx0 - 2*eper**2*jx0) + (-5 + 2*eper**2)*ex0*jz0)*Pi**4 - 2*(5*ez0*(jx0 + 4*eper**2*jx0) + (-25 + 4*eper**2)*ex0*jz0)*Pi**2*ArcCos(-(1/eper))**2 + 3*(ez0*(jx0 + 8*eper**2*jx0) + (-5 + 8*eper**2)*ex0*jz0)*ArcCos(-(1/eper))**4)*Sin((2*Pi*theta)/ArcCos(-(1/eper))))/f2 + ((-27*(ez0*(jx0 - 2*eper**2*jx0) + (-5 + 2*eper**2)*ex0*jz0)*Pi**4 - 3*(5*ez0*(jx0 + 4*eper**2*jx0) + (-25 + 4*eper**2)*ex0*jz0)*Pi**2*ArcCos(-(1/eper))**2 + 2*(ez0*(jx0 + 8*eper**2*jx0) + (-5 + 8*eper**2)*ex0*jz0)*ArcCos(-(1/eper))**4)*Sin((3*Pi*theta)/ArcCos(-(1/eper))))/f3))/(eper*Pi))
        ez = ez0 + eps_SA*(((3*(ey0*jx0 - ex0*jy0))/2. + (Sqrt(1 - eper**(-2))*((2 + eper**2)*ey0*jx0 + (2 - 5*eper**2)*ex0*jy0))/(2.*eper*ArcCos(-(1/eper))))*(theta + ArcCos(-(1/eper))) - (12*Sqrt(1 - eper**(-2))*(ex0*jx0 - ey0*jy0)*ArcCos(-(1/eper))**3*((-5 + 8*eper**2)*ArcCos(-(1/eper))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*eper**2)*Pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(eper*f1*f2*f3) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*((4*(ex0*jx0 - ey0*jy0)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper))))/f1 - (4*(ex0*jx0 - ey0*jy0)*Pi*ArcCos(-(1/eper))*(4*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((2*Pi*theta)/ArcCos(-(1/eper))))/f2 + (4*(ex0*jx0 - ey0*jy0)*Pi*ArcCos(-(1/eper))*(9*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((3*Pi*theta)/ArcCos(-(1/eper))))/f3 + ((((-4 + 3*eper**2)*ey0*jx0 + (-4 + eper**2)*ex0*jy0)*Pi**4 - (5*(4 + 3*eper**2)*ey0*jx0 + (20 - 11*eper**2)*ex0*jy0)*Pi**2*ArcCos(-(1/eper))**2 + 12*((2 + eper**2)*ey0*jx0 + (2 - 5*eper**2)*ex0*jy0)*ArcCos(-(1/eper))**4)*Sin((Pi*theta)/ArcCos(-(1/eper))))/f1 - (2*(4*((-4 + 3*eper**2)*ey0*jx0 + (-4 + eper**2)*ex0*jy0)*Pi**4 - (5*(4 + 3*eper**2)*ey0*jx0 + (20 - 11*eper**2)*ex0*jy0)*Pi**2*ArcCos(-(1/eper))**2 + 3*((2 + eper**2)*ey0*jx0 + (2 - 5*eper**2)*ex0*jy0)*ArcCos(-(1/eper))**4)*Sin((2*Pi*theta)/ArcCos(-(1/eper))))/f2 + ((27*((-4 + 3*eper**2)*ey0*jx0 + (-4 + eper**2)*ex0*jy0)*Pi**4 - 3*(5*(4 + 3*eper**2)*ey0*jx0 + (20 - 11*eper**2)*ex0*jy0)*Pi**2*ArcCos(-(1/eper))**2 + 4*((2 + eper**2)*ey0*jx0 + (2 - 5*eper**2)*ex0*jy0)*ArcCos(-(1/eper))**4)*Sin((3*Pi*theta)/ArcCos(-(1/eper))))/f3))/(eper*Pi))
        jx = jx0 + eps_SA*(-((5*ey0*ez0 - jy0*jz0)*(theta + ArcCos(-(1/eper)))*(Sqrt(1 - eper**(-2))*(1 + 2*eper**2) + 3*eper*ArcCos(-(1/eper))))/(4.*eper*ArcCos(-(1/eper))) + (3*Sqrt(1 - eper**(-2))*(5*ex0*ez0 - jx0*jz0)*ArcCos(-(1/eper))**3*((-5 + 8*eper**2)*ArcCos(-(1/eper))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*eper**2)*Pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(eper*f1*f2*f3) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*(-(((5*ex0*ez0 - jx0*jz0)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper))))/f1) + ((5*ex0*ez0 - jx0*jz0)*Pi*ArcCos(-(1/eper))*(4*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((2*Pi*theta)/ArcCos(-(1/eper))))/f2 - ((5*ex0*ez0 - jx0*jz0)*Pi*ArcCos(-(1/eper))*(9*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((3*Pi*theta)/ArcCos(-(1/eper))))/f3 - ((5*ey0*ez0 - jy0*jz0)*((-1 + eper**2)*Pi**4 - (5 + 7*eper**2)*Pi**2*ArcCos(-(1/eper))**2 + 6*(1 + 2*eper**2)*ArcCos(-(1/eper))**4)*Sin((Pi*theta)/ArcCos(-(1/eper))))/f1 + ((5*ey0*ez0 - jy0*jz0)*(8*(-1 + eper**2)*Pi**4 - 2*(5 + 7*eper**2)*Pi**2*ArcCos(-(1/eper))**2 + (3 + 6*eper**2)*ArcCos(-(1/eper))**4)*Sin((2*Pi*theta)/ArcCos(-(1/eper))))/f2 - ((5*ey0*ez0 - jy0*jz0)*(27*(-1 + eper**2)*Pi**4 - 3*(5 + 7*eper**2)*Pi**2*ArcCos(-(1/eper))**2 + (2 + 4*eper**2)*ArcCos(-(1/eper))**4)*Sin((3*Pi*theta)/ArcCos(-(1/eper))))/f3))/(eper*Pi))
        jy = jy0 + eps_SA*(((5*ex0*ez0 - jx0*jz0)*(theta + ArcCos(-(1/eper)))*(Sqrt(1 - eper**(-2))*(-1 + 4*eper**2) + 3*eper*ArcCos(-(1/eper))))/(4.*eper*ArcCos(-(1/eper))) - (3*Sqrt(1 - eper**(-2))*(5*ey0*ez0 - jy0*jz0)*ArcCos(-(1/eper))**3*((-5 + 8*eper**2)*ArcCos(-(1/eper))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*eper**2)*Pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(eper*f1*f2*f3) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*(((5*ey0*ez0 - jy0*jz0)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper))))/f1 - ((5*ey0*ez0 - jy0*jz0)*Pi*ArcCos(-(1/eper))*(4*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((2*Pi*theta)/ArcCos(-(1/eper))))/f2 + ((5*ey0*ez0 - jy0*jz0)*Pi*ArcCos(-(1/eper))*(9*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((3*Pi*theta)/ArcCos(-(1/eper))))/f3 + ((5*ex0*ez0 - jx0*jz0)*(Pi**4 + (5 - 6*eper**2)*Pi**2*ArcCos(-(1/eper))**2 + 6*(-1 + 4*eper**2)*ArcCos(-(1/eper))**4)*Sin((Pi*theta)/ArcCos(-(1/eper))))/f1 + ((5*ex0*ez0 - jx0*jz0)*(-8*Pi**4 + 2*(-5 + 6*eper**2)*Pi**2*ArcCos(-(1/eper))**2 + (3 - 12*eper**2)*ArcCos(-(1/eper))**4)*Sin((2*Pi*theta)/ArcCos(-(1/eper))))/f2 + ((5*ex0*ez0 - jx0*jz0)*(27*Pi**4 - 3*(-5 + 6*eper**2)*Pi**2*ArcCos(-(1/eper))**2 + (-2 + 8*eper**2)*ArcCos(-(1/eper))**4)*Sin((3*Pi*theta)/ArcCos(-(1/eper))))/f3))/(eper*Pi))
        jz = jz0 + eps_SA*(-(Sqrt(1 - eper**(-2))*(-1 + eper**2)*(5*ex0*ey0 - jx0*jy0)*(theta + ArcCos(-(1/eper))))/(2.*eper*ArcCos(-(1/eper))) - (3*Sqrt(1 - eper**(-2))*(5*ex0**2 - 5*ey0**2 - jx0**2 + jy0**2)*ArcCos(-(1/eper))**3*((-5 + 8*eper**2)*ArcCos(-(1/eper))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*eper**2)*Pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(eper*f1*f2*f3) + (3*Sqrt(1 - eper**(-2))*ArcCos(-(1/eper))**2*(((5*ex0**2 - 5*ey0**2 - jx0**2 + jy0**2)*Pi*ArcCos(-(1/eper))*((-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((Pi*theta)/ArcCos(-(1/eper))))/f1 - ((5*ex0**2 - 5*ey0**2 - jx0**2 + jy0**2)*Pi*ArcCos(-(1/eper))*(4*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((2*Pi*theta)/ArcCos(-(1/eper))))/f2 + ((5*ex0**2 - 5*ey0**2 - jx0**2 + jy0**2)*Pi*ArcCos(-(1/eper))*(9*(-5 + 2*eper**2)*Pi**2 + (5 - 8*eper**2)*ArcCos(-(1/eper))**2)*Cos((3*Pi*theta)/ArcCos(-(1/eper))))/f3 + ((5*ex0*ey0 - jx0*jy0)*((-2 + eper**2)*Pi**4 - (10 + eper**2)*Pi**2*ArcCos(-(1/eper))**2 - 12*(-1 + eper**2)*ArcCos(-(1/eper))**4)*Sin((Pi*theta)/ArcCos(-(1/eper))))/f1 - (2*(5*ex0*ey0 - jx0*jy0)*(4*(-2 + eper**2)*Pi**4 - (10 + eper**2)*Pi**2*ArcCos(-(1/eper))**2 - 3*(-1 + eper**2)*ArcCos(-(1/eper))**4)*Sin((2*Pi*theta)/ArcCos(-(1/eper))))/f2 + ((5*ex0*ey0 - jx0*jy0)*(27*(-2 + eper**2)*Pi**4 - 3*(10 + eper**2)*Pi**2*ArcCos(-(1/eper))**2 - 4*(-1 + eper**2)*ArcCos(-(1/eper))**4)*Sin((3*Pi*theta)/ArcCos(-(1/eper))))/f3))/(eper*Pi))
    return ex,ey,ez,jx,jy,jz

def compute_eps_SA(m,M_per,a_div_Q,e_per):
    return (M_per/np.sqrt(m*(m+M_per)))*pow(a_div_Q,3.0/2.0)*pow(1.0 + e_per,-3.0/2.0)
    
def compute_Q_div_a_from_eps_SA(eps_SA,m,M_per,e_per):
    return pow(eps_SA,-2.0/3.0)*pow( M_per**2/(m*(m+M_per)),1.0/3.0)*(1.0/(1.0+e_per))
    
def compute_eps_oct(m1,m2,m,a_div_Q,e_per):
    return a_div_Q*np.fabs(m1-m2)/((1.0+e_per)*m)
    
def compute_FO_prediction(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz):
    ArcCos = np.arccos
    Sqrt = np.sqrt
    eper = e_per
    Pi = np.pi

    epsilon = 1.0e-10
    if e_per <= 1.0 + epsilon:
        Delta_e = (15*ez*(ey*jx - ex*jy)*Pi*eps_SA)/(2.*Sqrt(ex**2 + ey**2 + ez**2))

        if args.include_octupole_terms==True:
            Delta_e += eps_oct*(-15*(ez*jy*(-4 - 3*ey**2 + 32*ez**2 + 5*jx**2 + 5*jy**2) + ey*(4 + 3*ey**2 - 32*ez**2 - 15*jx**2 - 5*jy**2)*jz + ex**2*(-73*ez*jy + 3*ey*jz) + 10*ex*jx*(7*ey*ez + jy*jz))*Pi*eps_SA)/(32.*Sqrt(ex**2 + ey**2 + ez**2))
    else:
        if args.use_c == True:
            Delta_e = flybyslibrary.f_e_cdot_e_hat(eps_SA,e_per,ex,ey,ez,jx,jy,jz)
        else:
            Delta_e = (5*eps_SA*(np.sqrt(1 - e_per**(-2))*((1 + 2*e_per**2)*ey*ez*jx + (1 - 4*e_per**2)*ex*ez*jy + 2*(-1 + e_per**2)*ex*ey*jz) + 3*e_per*ez*(ey*jx - ex*jy)*np.arccos(-1.0/e_per)))/(2.*e_per*np.sqrt(ex**2 + ey**2 + ez**2))

        if args.include_octupole_terms==True:
            Delta_e += -(5*eps_oct*eps_SA*(Sqrt(1 - eper**(-2))*(ez*jy*(14*ey**2 + 6*jx**2 - 2*jy**2 + 8*eper**4*(-1 + ey**2 + 8*ez**2 + 2*jx**2 + jy**2) + eper**2*(-4 - 31*ey**2 + 32*ez**2 - 7*jx**2 + 9*jy**2)) - ey*(2*(7*ey**2 + jx**2 - jy**2) + 8*eper**4*(-1 + ey**2 + 8*ez**2 + 4*jx**2 + jy**2) + eper**2*(-4 - 31*ey**2 + 32*ez**2 + 11*jx**2 + 9*jy**2))*jz + ex**2*(-((14 + 45*eper**2 + 160*eper**4)*ez*jy) + 3*(14 - 27*eper**2 + 16*eper**4)*ey*jz) + 2*(-2 + 9*eper**2 + 8*eper**4)*ex*jx*(7*ey*ez + jy*jz)) + 3*eper**3*(ez*jy*(-4 - 3*ey**2 + 32*ez**2 + 5*jx**2 + 5*jy**2) + ey*(4 + 3*ey**2 - 32*ez**2 - 15*jx**2 - 5*jy**2)*jz + ex**2*(-73*ez*jy + 3*ey*jz) + 10*ex*jx*(7*ey*ez + jy*jz))*ArcCos(-1.0/eper)))/(32.*eper**2*Sqrt(ex**2 + ey**2 + ez**2))
    

    Delta_i = -ArcCos(jz/Sqrt(jx**2 + jy**2 + jz**2)) + ArcCos((jz - (Sqrt(1 - eper**(-2))*(-1 + eper**2)*(5*ex*ey - jx*jy)*eps_SA)/eper)/Sqrt((jz - (Sqrt(1 - eper**(-2))*(-1 + eper**2)*(5*ex*ey - jx*jy)*eps_SA)/eper)**2 + (jx - ((5*ey*ez - jy*jz)*eps_SA*(Sqrt(1 - eper**(-2))*(1 + 2*eper**2) + 3*eper*ArcCos(-(1/eper))))/(2.*eper))**2 + (jy + ((5*ex*ez - jx*jz)*eps_SA*(Sqrt(1 - eper**(-2))*(-1 + 4*eper**2) + 3*eper*ArcCos(-(1/eper))))/(2.*eper))**2))
  
    return Delta_e,Delta_i


def f(n,e_per):
    L = np.arccos(-1.0/e_per)
    npi = n*np.pi
    return (npi - 3.0*L)*(npi - 2.0*L)*(npi - L)*(npi + L)*(npi + 2.0*L)*(npi + 3.0*L)

def compute_SO_prediction(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz):
    
    f1 = f(1.0,e_per)
    f2 = f(2.0,e_per)
    f3 = f(3.0,e_per)

    ArcCos = np.arccos
    Sqrt = np.sqrt
    eper = e_per
    Pi = np.pi

    epsilon = 1.0e-10
    if e_per <= 1.0 + epsilon: ### parabolic limit
#        The commented lines below correspond to an earlier, incorrect version of Delta_e in the parabolic limit. The correct expression (which gives numerically very similar results) is given below.
#        Delta_e_SO = eps_SA*(15*ez*(ey*jx - ex*jy)*np.pi)/(2.*np.sqrt(ex**2 + ey**2 + ez**2)) \
#            + eps_SA**2*(-9*(-6*ez**2*(jx**2 + jy**2) - 4*ey*ez*jy*jz + 4*ex*jx*(3*ey*jy - ez*jz) + ey**2*(25*ez**2 - 6*jx**2 + jz**2) + ex**2*(25*ez**2 - 6*jy**2 + jz**2))*np.pi**2)/(8.*np.sqrt(ex**2 + ey**2 + ez**2))
#        Delta_e_SO = eps_SA*(15*ez*(ey*jx - ex*jy)*np.pi)/(2.*np.sqrt(ex**2 + ey**2 + ez**2)) \
#            + (-3*Pi*(4*ey*ez*jz*(2*jx - 3*jy*Pi) - 2*ez**2*(jx*jy + 9*jx**2*Pi + 9*jy**2*Pi) + 3*ey**2*(-2*jx*jy + 25*ez**2*Pi - 6*jx**2*Pi + jz**2*Pi) + ex**2*(-6*jx*jy + 3*(25*ez**2 - 6*jy**2 + jz**2)*Pi) + ex*(4*ez*jz*(2*jy - 3*jx*Pi) + ey*(-25*ez**2 + 6*jx**2 + 6*jy**2 - 5*jz**2 + 36*jx*jy*Pi)))*eps_SA**2)/(8.*Sqrt(ex**2 + ey**2 + ez**2))

        ### First- and second-order eps_SA terms without octupole
        Delta_e_SO = eps_SA*(15*ez*(ey*jx - ex*jy)*np.pi)/(2.*np.sqrt(ex**2 + ey**2 + ez**2)) \
            + (3*Pi*(3*ex**2*(6*jy**2 - jz**2)*Pi + 2*ex*ez*jz*(-25*jy + 6*jx*Pi) + ez**2*(-25*jx*jy - 75*ex**2*Pi + 18*jx**2*Pi + 18*jy**2*Pi) + ey**2*(25*jx*jy + 18*jx**2*Pi - 3*(25*ez**2 + jz**2)*Pi) + ey*(2*ez*jz*(25*jx + 6*jy*Pi) + ex*(-25*jy**2 + 25*jz**2 - 36*jx*jy*Pi)))*eps_SA**2)/(8.*Sqrt(ex**2 + ey**2 + ez**2))

        if args.include_octupole_terms==True:
            ### First-order eps_SA octupole term
            Delta_e_SO += eps_oct*(-15*(ez*jy*(-4 - 3*ey**2 + 32*ez**2 + 5*jx**2 + 5*jy**2) + ey*(4 + 3*ey**2 - 32*ez**2 - 15*jx**2 - 5*jy**2)*jz + ex**2*(-73*ez*jy + 3*ey*jz) + 10*ex*jx*(7*ey*ez + jy*jz))*Pi*eps_SA)/(32.*Sqrt(ex**2 + ey**2 + ez**2))

            ### Second-order eps_SA octupole terms
            Delta_e_SO += eps_oct*(-15*Pi*(3*ex**2*ez*jz*(-3759*jy + 1100*jx*Pi) + ey**3*(-707*jx**2 + 2037*jz**2 + 12*jx*jy*Pi) + 3*ez*jz*(7*(-20 + 160*ez**2 + 51*jx**2)*jy + 301*jy**3 - 4*jx*(-12 + 96*ez**2 + 35*jx**2)*Pi - 140*jx*jy**2*Pi) - ex**3*(847*jx*jy + 60*(146*ez**2 - 33*jy**2 + 3*jz**2)*Pi) + ex*(-49*jx**3*jy + 7*jx*jy*(20 - 676*ez**2 + jy**2 - 62*jz**2) + 12*jx**2*(276*ez**2 + 5*(-3*jy**2 + jz**2))*Pi + 12*(320*ez**4 - 15*jy**4 - 4*jz**2 + 3*jy**2*(4 + 5*jz**2) + 4*ez**2*(-10 + 65*jy**2 + 8*jz**2))*Pi) + ey*(49*jx**4 + 21*(20 - 43*jy**2)*jz**2 + 7*jx**2*(-20 + 121*ex**2 + 682*ez**2 - jy**2 - 91*jz**2) + 42*ez**2*(73*jy**2 - 80*jz**2) + ex**2*(-3066*jy**2 + 3549*jz**2) + 180*jx**3*jy*Pi + 2712*ex*ez*jy*jz*Pi - 6*jx*(-1281*ex*ez*jz + 662*ex**2*jy*Pi + 2*jy*(12 - 16*ez**2 - 15*jy**2 + 10*jz**2)*Pi)) + ey**2*(21*ez*jz*(-243*jy + 28*jx*Pi) + ex*(3773*jx*jy + 1992*jx**2*Pi - 12*(730*ez**2 + jy**2 + 15*jz**2)*Pi)))*eps_SA**2)/(512.*Sqrt(ex**2 + ey**2 + ez**2))
            Delta_e_SO += eps_oct**2*(-225*Pi*(36*ey**6*Pi + 2*ey**3*ez*jz*(-4067*jx + 992*jy*Pi) + 2*ey*ez*jz*(5517*jx**3 + jx*(-2408 + 19264*ez**2 + 3945*jy**2) - 240*jx**2*jy*Pi + 16*jy*(-34 + 272*ez**2 + 55*jy**2)*Pi) + 4*ez**2*(519*jx**3*jy + 3*jx*jy*(-28 + 224*ez**2 + 191*jy**2) + 75*jx**4*Pi + 10*jx**2*(-8 + 64*ez**2 - 17*jy**2)*Pi + (16 + 1024*ez**4 + 176*jy**2 - 245*jy**4 - 128*ez**2*(2 + 11*jy**2))*Pi) - ey**4*(7679*jx*jy + 240*jx**2*Pi + 4*(-24 + 183*ez**2 - 10*jy**2 + 18*jz**2)*Pi) + ex**4*(7343*jx*jy + 4*(9*ey**2 + 5329*ez**2 + 54*(-19*jy**2 + jz**2))*Pi) + ey**2*(-1795*jx**3*jy - jx*jy*(-1008 + 3374*ez**2 + 55*jy**2 + 3802*jz**2) + 300*jx**4*Pi - 40*jx**2*(8 + 2*ez**2 - 30*jy**2 - 39*jz**2)*Pi + 4*(16 + 832*ez**4 + 125*jy**4 - 24*jz**2 - 10*jy**2*(12 + 7*jz**2) + 4*ez**2*(-58 + 123*jy**2 + 48*jz**2))*Pi) + ex**2*(-281*jx**3*jy - jx*(2237*jy**3 + 14350*ey*ez*jz + 2*jy*(336 + 6048*ey**2 - 4207*ez**2 - 1901*jz**2)) - 8*jx**2*(602*ey**2 + 1342*ez**2 - 85*jy**2 - 105*jz**2)*Pi + 8*(9*ey**4 - 2336*ez**4 - 108*jy**2 + 135*jy**4 - 608*ey*ez*jy*jz + 12*jz**2 - 65*jy**2*jz**2 + ey**2*(12 + 2573*ez**2 + 64*jy**2 + 18*jz**2) - 2*ez**2*(-146 + 871*jy**2 + 48*jz**2))*Pi) + ex**3*(10*ez*jz*(2191*jy - 1216*jx*Pi) - 7*ey*(1049*jx**2 - 2045*jy**2 + 612*jz**2 - 1280*jx*jy*Pi)) + ex*(2*ey**2*ez*jz*(9359*jy - 2656*jx*Pi) - ey**3*(2219*jx**2 - 7679*jy**2 + 2772*jz**2 + 192*jx*jy*Pi) + 2*ez*jz*((3752 - 30016*ez**2 - 8817*jx**2)*jy - 7029*jy**3 + 16*jx*(-38 + 304*ez**2 + 105*jx**2)*Pi + 2800*jx*jy**2*Pi) + ey*(281*jx**4 + 55*jy**4 + 2352*(-1 + 8*ez**2)*jz**2 - 2*jy**2*(504 + 2219*ez**2 - 3839*jz**2) + jx**2*(672 - 11690*ez**2 + 4032*jy**2 + 722*jz**2) - 480*jx**3*jy*Pi - 32*jx*jy*(-22 - 164*ez**2 + 40*jy**2 + 15*jz**2)*Pi)))*eps_SA**2)/(8192.*Sqrt(ex**2 + ey**2 + ez**2))

        Delta_i =  -ArcCos(jz/Sqrt(jx**2 + jy**2 + jz**2)) + ArcCos((8*jz)/Sqrt(64*jz**2 + (8*jx - 3*Pi*eps_SA*(20*ey*ez - 4*jy*jz + 3*(10*ey**2*jx + 15*ez**2*jx - 10*ex*ey*jy + jx*jz**2)*Pi*eps_SA))**2 + (8*jy - 3*Pi*eps_SA*(-20*ex*ez + 4*jx*jz + 3*(-10*ex*ey*jx + 10*ex**2*jy + 15*ez**2*jy + jy*jz**2)*Pi*eps_SA))**2))

    else: ### hyperbolic orbits
        Delta_i = -ArcCos(jz/Sqrt(jx**2 + jy**2 + jz**2)) + ArcCos((jz - (Sqrt(1 - eper**(-2))*(-1 + eper**2)*(5*ex*ey - jx*jy)*eps_SA)/eper - (Sqrt(1 - eper**(-2))*(-1 + eper**2)*eps_SA**2*(Sqrt(1 - eper**(-2))*(10*(1 + 2*eper**2)*ex*ez*jx + 10*(1 - 4*eper**2)*ey*ez*jy + 5*(-5 + 8*eper**2)*ex**2*jz + 5*(-5 + 2*eper**2)*ey**2*jz + (-1 + 4*eper**2)*jx**2*jz - (1 + 2*eper**2)*jy**2*jz) - 3*eper*(5*ex*ez*jx - 5*ey*ez*jy + (-jx**2 + jy**2)*jz)*ArcCos(-(1/eper)) + 15*eper*(3*ex*ez*jx + ex**2*jz - ey*(3*ez*jy + ey*jz))*ArcCos(-1.0/eper)))/(4.*eper**2))/Sqrt((jz - (Sqrt(1 - eper**(-2))*(-1 + eper**2)*(5*ex*ey - jx*jy)*eps_SA)/eper - (Sqrt(1 - eper**(-2))*(-1 + eper**2)*eps_SA**2*(Sqrt(1 - eper**(-2))*(10*(1 + 2*eper**2)*ex*ez*jx + 10*(1 - 4*eper**2)*ey*ez*jy + 5*(-5 + 8*eper**2)*ex**2*jz + 5*(-5 + 2*eper**2)*ey**2*jz + (-1 + 4*eper**2)*jx**2*jz - (1 + 2*eper**2)*jy**2*jz) - 3*eper*(5*ex*ez*jx - 5*ey*ez*jy + (-jx**2 + jy**2)*jz)*ArcCos(-(1/eper)) + 15*eper*(3*ex*ez*jx + ex**2*jz - ey*(3*ez*jy + ey*jz))*ArcCos(-1.0/eper)))/(4.*eper**2))**2 + (jy + ((5*ex*ez - jx*jz)*eps_SA*(Sqrt(1 - eper**(-2))*(-1 + 4*eper**2) + 3*eper*ArcCos(-(1/eper))))/(2.*eper) - (eps_SA**2*(Sqrt(1 - eper**(-2))*(-1 + 4*eper**2) + 3*eper*ArcCos(-(1/eper)))*(3*eper*(-10*ex*ey*jx + 10*ex**2*jy + jz*(-5*ey*ez + jy*jz))*ArcCos(-(1/eper))*f1*f2*f3 + (Sqrt(1 - eper**(-2))*(-10*(1 + 2*eper**2)*ex*ey*jx + 10*(-2 + 5*eper**2)*ex**2*jy + 5*(-1 + 10*eper**2)*ez**2*jy + 2*(-1 + eper**2)*jx**2*jy - 20*(-1 + eper**2)*ey*ez*jz + (1 + 2*eper**2)*jy*jz**2) + 15*eper*ez*(3*ez*jy + ey*jz)*ArcCos(-1.0/eper))*f1*f2*f3 + 12*Sqrt(1 - eper**(-2))*(-5 + 8*eper**2)*(15*ex**2*jx - 20*ex*(ey*jy + ez*jz) + jx*(5*ey**2 + 5*ez**2 + jx**2 - jy**2 - jz**2))*ArcCos(-(1/eper))**5*(f2*f3 + f1*(f2 + f3)) - 12*Sqrt(1 - eper**(-2))*(-5 + 2*eper**2)*(15*ex**2*jx - 20*ex*(ey*jy + ez*jz) + jx*(5*ey**2 + 5*ez**2 + jx**2 - jy**2 - jz**2))*Pi**2*ArcCos(-(1/eper))**3*(f2*f3 + f1*(9*f2 + 4*f3))))/(8.*eper**2*f1*f2*f3))**2 + (jx - ((5*ey*ez - jy*jz)*eps_SA*(Sqrt(1 - eper**(-2))*(1 + 2*eper**2) + 3*eper*ArcCos(-(1/eper))))/(2.*eper) - (eps_SA**2*(Sqrt(1 - eper**(-2))*(1 + 2*eper**2) + 3*eper*ArcCos(-(1/eper)))*(3*eper*(10*ey**2*jx - 10*ex*ey*jy + jz*(-5*ex*ez + jx*jz))*ArcCos(-(1/eper))*f1*f2*f3 + (Sqrt(1 - eper**(-2))*(10*(2 + eper**2)*ey**2*jx + 5*ez**2*(jx + 8*eper**2*jx) + 10*(1 - 4*eper**2)*ex*ey*jy + 20*(-1 + eper**2)*ex*ez*jz - jx*(2*(-1 + eper**2)*jy**2 + (1 - 4*eper**2)*jz**2)) + 15*eper*ez*(3*ez*jx + ex*jz)*ArcCos(-1.0/eper))*f1*f2*f3 + 12*Sqrt(1 - eper**(-2))*(-5 + 8*eper**2)*(-20*ex*ey*jx + 5*ex**2*jy + 15*ey**2*jy - 20*ey*ez*jz + jy*(5*ez**2 - jx**2 + jy**2 - jz**2))*ArcCos(-(1/eper))**5*(f2*f3 + f1*(f2 + f3)) - 12*Sqrt(1 - eper**(-2))*(-5 + 2*eper**2)*(-20*ex*ey*jx + 5*ex**2*jy + 15*ey**2*jy - 20*ey*ez*jz + jy*(5*ez**2 - jx**2 + jy**2 - jz**2))*Pi**2*ArcCos(-(1/eper))**3*(f2*f3 + f1*(9*f2 + 4*f3))))/(8.*eper**2*f1*f2*f3))**2))

        if args.use_c == True:
            Delta_e_SO = flybyslibrary.g_e_I_cdot_e_hat(eps_SA,e_per,ex,ey,ez,jx,jy,jz)
            Delta_e_SO += flybyslibrary.g_e_II_cdot_e_hat(eps_SA,e_per,ex,ey,ez,jx,jy,jz)
        else:
            exn = ex + (eps_SA*(np.sqrt(1 - e_per**(-2))*(ez*(jy - 10*e_per**2*jy) + (-5 + 2*e_per**2)*ey*jz) - 3*e_per*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per)))/(4.*e_per) - (3*np.sqrt(1 - e_per**(-2))*(ez*jx - 5*ex*jz)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)
            eyn = ey + (eps_SA*(np.sqrt(1 - e_per**(-2))*(ez*(jx + 8*e_per**2*jx) + (-5 + 8*e_per**2)*ex*jz) + 3*e_per*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per)))/(4.*e_per) + (3*np.sqrt(1 - e_per**(-2))*(ez*jy - 5*ey*jz)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)
            ezn = ez + (np.sqrt(1 - e_per**(-2))*((2 + e_per**2)*ey*jx + (2 - 5*e_per**2)*ex*jy)*eps_SA + 3*e_per*(ey*jx - ex*jy)*eps_SA*np.arccos(-(1/e_per)))/(2.*e_per) - (12*np.sqrt(1 - e_per**(-2))*(ex*jx - ey*jy)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)
            jxn = jx - ((5*ey*ez - jy*jz)*eps_SA*(np.sqrt(1 - e_per**(-2))*(1 + 2*e_per**2) + 3*e_per*np.arccos(-(1/e_per))))/(4.*e_per) + (3*np.sqrt(1 - e_per**(-2))*(5*ex*ez - jx*jz)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)
            jyn = jy + ((5*ex*ez - jx*jz)*eps_SA*(np.sqrt(1 - e_per**(-2))*(-1 + 4*e_per**2) + 3*e_per*np.arccos(-(1/e_per))))/(4.*e_per) - (3*np.sqrt(1 - e_per**(-2))*(5*ey*ez - jy*jz)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)
            jzn = jz - (np.sqrt(1 - e_per**(-2))*(-1 + e_per**2)*(5*ex*ey - jx*jy)*eps_SA)/(2.*e_per) - (3*np.sqrt(1 - e_per**(-2))*(5*ex**2 - 5*ey**2 - jx**2 + jy**2)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)

            Delta_e_SO = compute_FO_prediction(args,eps_SA,0.0,e_per,exn,eyn,ezn,jxn,jyn,jzn)[0]

            lmax=3
            if lmax==1:
                Delta_e_SO += (3*eps_SA**2*np.arccos(-(1/e_per))*((5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(60*(-1 + e_per**2)*((-1 + e_per**2)*ex**2*jx*jy + ey**2*jx*jy - e_per**2*ez**2*jx*jy - 2*(-2 + e_per**2)*ey*ez*jx*jz + 2*(-2 + e_per**2)*ex*ez*jy*jz + ex*ey*(-((-1 + e_per**2)*jx**2) - jy**2 + e_per**2*jz**2))*np.pi**4*np.arccos(-(1/e_per))**3 - 60*(-1 + e_per**2)*((5 + 7*e_per**2)*ex**2*jx*jy + (-5 + 6*e_per**2)*ey**2*jx*jy - 13*e_per**2*ez**2*jx*jy - 2*(10 + e_per**2)*ey*ez*jx*jz + 2*(10 + e_per**2)*ex*ez*jy*jz + ex*ey*(-((5 + 7*e_per**2)*jx**2) + (5 - 6*e_per**2)*jy**2 + 13*e_per**2*jz**2))*np.pi**2*np.arccos(-(1/e_per))**5 + 360*(-1 + e_per**2)*((1 + 2*e_per**2)*ex**2*jx*jy + (-1 + 4*e_per**2)*ey**2*jx*jy - 6*e_per**2*ez**2*jx*jy + 4*(-1 + e_per**2)*ey*ez*jx*jz - 4*(-1 + e_per**2)*ex*ez*jy*jz + ex*ey*(-((1 + 2*e_per**2)*jx**2) + (1 - 4*e_per**2)*jy**2 + 6*e_per**2*jz**2))*np.arccos(-(1/e_per))**7 - 3*np.sqrt(1 - e_per**(-2))*e_per**3*(2*ex**2*jx*jy + ey*jx*(2*ey*jy - 5*ez*jz) + ex*(50*ey*ez**2 - 2*ey*(jx**2 + jy**2) - 5*ez*jy*jz))*np.arccos(-(1/e_per))*np.arccos(-1.0/e_per)*f1 + np.arccos(-1.0/e_per)*(2*(-1 + e_per**2)*(25*(-1 + e_per**2)*ex**3*ey + (7 - 10*e_per**2)*ex**2*jx*jy + jx*((-7 + 4*e_per**2)*ey**2*jy - 36*e_per**2*ez**2*jy + 2*(-5 + 17*e_per**2)*ey*ez*jz) + ex*(-25*(-1 + e_per**2)*ey**3 + 2*ey*(jx**2 - jy**2) + 2*(5 + 7*e_per**2)*ez*jy*jz + e_per**2*ey*(-75*ez**2 + jx**2 + 5*jy**2 + 15*jz**2))) - 3*np.sqrt(1 - e_per**(-2))*e_per**3*(24*ez**2*jx*jy - 11*ez*(ey*jx + ex*jy)*jz - 10*ex*ey*jz**2)*np.arccos(-1.0/e_per))*f1))/(2.*e_per**4*np.sqrt(ex**2 + ey**2 + ez**2)*f1**2)
            if lmax==2:
                Delta_e_SO += (3*eps_SA**2*np.arccos(-(1/e_per))*np.arccos(-1.0/e_per)*((ey*(-12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*(-8*(2*jx*((7 - 4*e_per**2)*ey*jy + (-2 + 5*e_per**2)*ez*jz) + ex*(25*(-2 + e_per**2)*ey**2 - 5*ez**2 + 4*jy**2 - e_per**2*jy**2 + 25*jz**2 - 10*e_per**2*jz**2))*np.pi**4 + 2*(2*jx*(-5*(7 + 2*e_per**2)*ey*jy + (10 + 53*e_per**2)*ez*jz) + ex*(25*(10 + e_per**2)*ey**2 - 5*(-5 + 6*e_per**2)*ez**2 - 20*jy**2 + 11*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 3*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.arccos(-(1/e_per))**4)*f1**2 - 12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(-8*(5*(-5 + 2*e_per**2)*ex**3 + 4*jx*((-1 + 2*e_per**2)*ey*jy - ez*jz) + ex*(-5*(-5 + 2*e_per**2)*ey**2 + 5*(-1 + 2*e_per**2)*ez**2 + 9*jx**2 - 10*e_per**2*jx**2 - 5*jy**2 + 2*e_per**2*jy**2 + 25*jz**2 - 10*e_per**2*jz**2))*np.pi**5 + 2*(-5*(-25 + 4*e_per**2)*ex**3 + 4*jx*(5*ey*(jy + 4*e_per**2*jy) + (5 - 6*e_per**2)*ez*jz) + ex*(5*(-25 + 4*e_per**2)*ey**2 + 25*(1 + 4*e_per**2)*ez**2 - 45*jx**2 - 76*e_per**2*jx**2 + 25*jy**2 - 4*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 3*(5*(-5 + 8*e_per**2)*ex**3 - 4*jx*(ey*(jy + 8*e_per**2*jy) + (1 - 4*e_per**2)*ez*jz) + ex*(-5*(-5 + 8*e_per**2)*ey**2 - 5*(1 + 8*e_per**2)*ez**2 + 9*jx**2 + 24*e_per**2*jx**2 - 5*jy**2 + 8*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f1**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*(4*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2)) + 3*e_per*(5*ex*ez**2 + 2*ey*jx*jy - 2*ex*jy**2 - ez*jx*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per))*f1**2*f2 - 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*((2*jx*((-7 + 4*e_per**2)*ey*jy + (2 - 5*e_per**2)*ez*jz) + ex*(-25*(-2 + e_per**2)*ey**2 + 5*ez**2 - 4*jy**2 + e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi**4 + (2*jx*(-5*(7 + 2*e_per**2)*ey*jy + (10 + 53*e_per**2)*ez*jz) + ex*(25*(10 + e_per**2)*ey**2 - 5*(-5 + 6*e_per**2)*ez**2 - 20*jy**2 + 11*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 6*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.arccos(-(1/e_per))**4)*f2**2 - 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*((-5*(-5 + 2*e_per**2)*ex**3 + 4*jx*(ey*(jy - 2*e_per**2*jy) + ez*jz) + ex*(5*(-5 + 2*e_per**2)*ey**2 + (5 - 10*e_per**2)*ez**2 - 9*jx**2 + 10*e_per**2*jx**2 + 5*jy**2 - 2*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi**5 + (-5*(-25 + 4*e_per**2)*ex**3 + 4*jx*(5*ey*(jy + 4*e_per**2*jy) + (5 - 6*e_per**2)*ez*jz) + ex*(5*(-25 + 4*e_per**2)*ey**2 + 25*(1 + 4*e_per**2)*ez**2 - 45*jx**2 - 76*e_per**2*jx**2 + 25*jy**2 - 4*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 6*(5*(-5 + 8*e_per**2)*ex**3 - 4*jx*(ey*(jy + 8*e_per**2*jy) + (1 - 4*e_per**2)*ez*jz) + ex*(-5*(-5 + 8*e_per**2)*ey**2 - 5*(1 + 8*e_per**2)*ez**2 + 9*jx**2 + 24*e_per**2*jx**2 - 5*jy**2 + 8*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f2**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*((-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2)) + 3*e_per*(5*ex*ez**2 + 2*ey*jx*jy - 2*ex*jy**2 - ez*jx*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per))*f1*f2**2))/np.pi + (ex*(12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*(-8*(25*(-2 + e_per**2)*ex**2*ey + 2*(7 - 3*e_per**2)*ex*jx*jy - 2*(2 + 3*e_per**2)*ez*jy*jz + ey*(5*(-1 + e_per**2)*ez**2 + (4 - 3*e_per**2)*jx**2 + 5*(5 - 3*e_per**2)*jz**2))*np.pi**4 + 2*(25*(10 + e_per**2)*ex**2*ey + 2*(-35 + 3*e_per**2)*ex*jx*jy + 2*(10 - 51*e_per**2)*ez*jy*jz + 5*ey*((5 + 7*e_per**2)*ez**2 - (4 + 3*e_per**2)*jx**2 - (25 + 9*e_per**2)*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 3*(50*(-1 + e_per**2)*ex**2*ey + 2*(7 - 10*e_per**2)*ex*jx*jy + 4*(-1 + 13*e_per**2)*ez*jy*jz + ey*(-5*(1 + 2*e_per**2)*ez**2 + 2*(2 + e_per**2)*jx**2 + 5*(5 - 2*e_per**2)*jz**2))*np.arccos(-(1/e_per))**4)*f1**2 - 12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(-8*(5*(-5 + 3*e_per**2)*ex**2*ey - 5*(-5 + 3*e_per**2)*ey**3 + 4*(1 + e_per**2)*ex*jx*jy - 4*(-1 + e_per**2)*ez*jy*jz + ey*(5*(1 + e_per**2)*ez**2 + (5 - 3*e_per**2)*jx**2 - 9*jy**2 - e_per**2*jy**2 - 25*jz**2 + 15*e_per**2*jz**2))*np.pi**5 + 2*(5*(25 + 9*e_per**2)*ex**2*ey - 5*(25 + 9*e_per**2)*ey**3 + 4*(-5 + 19*e_per**2)*ex*jx*jy - 4*(5 + 7*e_per**2)*ez*jy*jz + ey*(5*(-5 + 19*e_per**2)*ez**2 - (25 + 9*e_per**2)*jx**2 + 45*jy**2 - 67*e_per**2*jy**2 + 125*jz**2 + 45*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 3*(5*(-5 + 2*e_per**2)*ex**2*ey - 5*(-5 + 2*e_per**2)*ey**3 + 4*(1 - 10*e_per**2)*ex*jx*jy + 4*(1 + 2*e_per**2)*ez*jy*jz + ey*((5 - 50*e_per**2)*ez**2 + (5 - 2*e_per**2)*jx**2 - 9*jy**2 + 42*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f1**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*(4*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-50*(-1 + e_per**2)*ex**2*ey + 2*(-7 + 10*e_per**2)*ex*jx*jy + 4*(1 - 13*e_per**2)*ez*jy*jz + ey*(5*(1 + 2*e_per**2)*ez**2 - 2*(2 + e_per**2)*jx**2 + 5*(-5 + 2*e_per**2)*jz**2)) + 3*e_per*(5*ey*ez**2 - 2*ey*jx**2 + 2*ex*jx*jy - ez*jy*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per))*f1**2*f2 + 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*((-25*(-2 + e_per**2)*ex**2*ey + 2*(-7 + 3*e_per**2)*ex*jx*jy + 2*(2 + 3*e_per**2)*ez*jy*jz + ey*(-5*(-1 + e_per**2)*ez**2 + (-4 + 3*e_per**2)*jx**2 + 5*(-5 + 3*e_per**2)*jz**2))*np.pi**4 + (25*(10 + e_per**2)*ex**2*ey + 2*(-35 + 3*e_per**2)*ex*jx*jy + 2*(10 - 51*e_per**2)*ez*jy*jz + 5*ey*((5 + 7*e_per**2)*ez**2 - (4 + 3*e_per**2)*jx**2 - (25 + 9*e_per**2)*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 6*(50*(-1 + e_per**2)*ex**2*ey + 2*(7 - 10*e_per**2)*ex*jx*jy + 4*(-1 + 13*e_per**2)*ez*jy*jz + ey*(-5*(1 + 2*e_per**2)*ez**2 + 2*(2 + e_per**2)*jx**2 + 5*(5 - 2*e_per**2)*jz**2))*np.arccos(-(1/e_per))**4)*f2**2 - 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*((-5*(-5 + 3*e_per**2)*ex**2*ey + 5*(-5 + 3*e_per**2)*ey**3 - 4*(1 + e_per**2)*ex*jx*jy + 4*(-1 + e_per**2)*ez*jy*jz + ey*(-5*(1 + e_per**2)*ez**2 + (-5 + 3*e_per**2)*jx**2 + 9*jy**2 + e_per**2*jy**2 + 25*jz**2 - 15*e_per**2*jz**2))*np.pi**5 + (5*(25 + 9*e_per**2)*ex**2*ey - 5*(25 + 9*e_per**2)*ey**3 + 4*(-5 + 19*e_per**2)*ex*jx*jy - 4*(5 + 7*e_per**2)*ez*jy*jz + ey*(5*(-5 + 19*e_per**2)*ez**2 - (25 + 9*e_per**2)*jx**2 + 45*jy**2 - 67*e_per**2*jy**2 + 125*jz**2 + 45*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 6*(5*(-5 + 2*e_per**2)*ex**2*ey - 5*(-5 + 2*e_per**2)*ey**3 + 4*(1 - 10*e_per**2)*ex*jx*jy + 4*(1 + 2*e_per**2)*ez*jy*jz + ey*((5 - 50*e_per**2)*ez**2 + (5 - 2*e_per**2)*jx**2 - 9*jy**2 + 42*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f2**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*((-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-50*(-1 + e_per**2)*ex**2*ey + 2*(-7 + 10*e_per**2)*ex*jx*jy + 4*(1 - 13*e_per**2)*ez*jy*jz + ey*(5*(1 + 2*e_per**2)*ez**2 - 2*(2 + e_per**2)*jx**2 + 5*(-5 + 2*e_per**2)*jz**2)) + 3*e_per*(5*ey*ez**2 - 2*ey*jx**2 + 2*ex*jx*jy - ez*jy*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per))*f1*f2**2))/np.pi - 12*np.sqrt(1 - e_per**(-2))*e_per**2*ez*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*np.arccos(-1.0/e_per)*(2*np.sqrt(1 - e_per**(-2))*((ey*jx - ex*jy)*jz + e_per**2*(5*ex*ey*ez + 3*ez*jx*jy - ey*jx*jz + ex*jy*jz)) + e_per*(6*ez*jx*jy + ey*jx*jz + ex*jy*jz)*np.arccos(-1.0/e_per))*f1*f2*(f1 + f2) - e_per*(-5 + 2*e_per**2)*(10*ex*ey*ez - ey*jx*jz - ex*jy*jz)*np.pi**2*np.arccos(-(1/e_per))*np.arccos(-1.0/e_per)*f1*f2*(4*f1 + f2) - (-5 + 2*e_per**2)*np.pi**2*np.arccos(-1.0/e_per)*(2*np.sqrt(1 - e_per**(-2))*((ey*jx - ex*jy)*jz + e_per**2*(5*ex*ey*ez + 3*ez*jx*jy - ey*jx*jz + ex*jy*jz)) + e_per*(6*ez*jx*jy + ey*jx*jz + ex*jy*jz)*np.arccos(-1.0/e_per))*f1*f2*(4*f1 + f2) + 36*np.sqrt(1 - e_per**(-2))*(-5 + 8*e_per**2)*(4*(ey*jx - ex*jy)*jz + e_per**2*(5*ex*ey*ez + 5*ez*jx*jy - ey*jx*jz + 7*ex*jy*jz))*np.arccos(-(1/e_per))**9*(f1**2 + f2**2) - np.sqrt(1 - e_per**(-2))*(1320*(-(ey*jx*jz) + ex*jy*jz) + 16*e_per**4*(55*ex*ey*ez + 55*ez*jx*jy + 21*ey*jx*jz + 45*ex*jy*jz) - e_per**2*(1225*ex*ey*ez + 1225*ez*jx*jy - 1173*ey*jx*jz + 2643*ex*jy*jz))*np.pi**2*np.arccos(-(1/e_per))**7*(4*f1**2 + f2**2) + 2*np.sqrt(1 - e_per**(-2))*(e_per**4*(85*ex*ey*ez + 85*ez*jx*jy + 111*ey*jx*jz - 9*ex*jy*jz) + 240*(-(ey*jx*jz) + ex*jy*jz) - e_per**2*(175*ex*ey*ez + 175*ez*jx*jy + 141*ey*jx*jz + 69*ex*jy*jz))*np.pi**4*np.arccos(-(1/e_per))**5*(16*f1**2 + f2**2) + np.arccos(-(1/e_per))**3*(e_per*(-5 + 8*e_per**2)*(10*ex*ey*ez - ey*jx*jz - ex*jy*jz)*np.arccos(-1.0/e_per)*f1*f2*(f1 + f2) - np.sqrt(1 - e_per**(-2))*(-5 + 2*e_per**2)*(24*(-(ey*jx) + ex*jy)*jz + e_per**2*(5*ex*ey*ez + 5*ez*jx*jy + 15*ey*jx*jz - 9*ex*jy*jz))*np.pi**6*(64*f1**2 + f2**2)))))/(2.*e_per**4*np.sqrt(ex**2 + ey**2 + ez**2)*f1**2*f2**2)
            if lmax==3:
                Delta_e_SO += (3*eps_SA**2*np.arccos(-(1/e_per))*np.arccos(-1.0/e_per)*((ey*(-18*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(9*(5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*(-27*(2*jx*((7 - 4*e_per**2)*ey*jy + (-2 + 5*e_per**2)*ez*jz) + ex*(25*(-2 + e_per**2)*ey**2 - 5*ez**2 + 4*jy**2 - e_per**2*jy**2 + 25*jz**2 - 10*e_per**2*jz**2))*np.pi**4 + 3*(2*jx*(-5*(7 + 2*e_per**2)*ey*jy + (10 + 53*e_per**2)*ez*jz) + ex*(25*(10 + e_per**2)*ey**2 - 5*(-5 + 6*e_per**2)*ez**2 - 20*jy**2 + 11*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 2*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.arccos(-(1/e_per))**4)*f1**2*f2**2 - 18*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(9*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(-27*(5*(-5 + 2*e_per**2)*ex**3 + 4*jx*((-1 + 2*e_per**2)*ey*jy - ez*jz) + ex*(-5*(-5 + 2*e_per**2)*ey**2 + 5*(-1 + 2*e_per**2)*ez**2 + 9*jx**2 - 10*e_per**2*jx**2 - 5*jy**2 + 2*e_per**2*jy**2 + 25*jz**2 - 10*e_per**2*jz**2))*np.pi**5 - 3*(5*(-25 + 4*e_per**2)*ex**3 - 4*jx*(5*ey*(jy + 4*e_per**2*jy) + (5 - 6*e_per**2)*ez*jz) + ex*(-5*(-25 + 4*e_per**2)*ey**2 - 25*(1 + 4*e_per**2)*ez**2 + 45*jx**2 + 76*e_per**2*jx**2 - 25*jy**2 + 4*e_per**2*jy**2 + 125*jz**2 - 20*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 2*(5*(-5 + 8*e_per**2)*ex**3 - 4*jx*(ey*(jy + 8*e_per**2*jy) + (1 - 4*e_per**2)*ez*jz) + ex*(-5*(-5 + 8*e_per**2)*ey**2 - 5*(1 + 8*e_per**2)*ez**2 + 9*jx**2 + 24*e_per**2*jx**2 - 5*jy**2 + 8*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f1**2*f2**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*(9*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2)) + 3*e_per*(5*ex*ez**2 + 2*ey*jx*jy - 2*ex*jy**2 - ez*jx*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per))*f1**2*f2**2*f3 - 12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*(-8*(2*jx*((7 - 4*e_per**2)*ey*jy + (-2 + 5*e_per**2)*ez*jz) + ex*(25*(-2 + e_per**2)*ey**2 - 5*ez**2 + 4*jy**2 - e_per**2*jy**2 + 25*jz**2 - 10*e_per**2*jz**2))*np.pi**4 + 2*(2*jx*(-5*(7 + 2*e_per**2)*ey*jy + (10 + 53*e_per**2)*ez*jz) + ex*(25*(10 + e_per**2)*ey**2 - 5*(-5 + 6*e_per**2)*ez**2 - 20*jy**2 + 11*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 3*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.arccos(-(1/e_per))**4)*f1**2*f3**2 - 12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(-8*(5*(-5 + 2*e_per**2)*ex**3 + 4*jx*((-1 + 2*e_per**2)*ey*jy - ez*jz) + ex*(-5*(-5 + 2*e_per**2)*ey**2 + 5*(-1 + 2*e_per**2)*ez**2 + 9*jx**2 - 10*e_per**2*jx**2 - 5*jy**2 + 2*e_per**2*jy**2 + 25*jz**2 - 10*e_per**2*jz**2))*np.pi**5 + 2*(-5*(-25 + 4*e_per**2)*ex**3 + 4*jx*(5*ey*(jy + 4*e_per**2*jy) + (5 - 6*e_per**2)*ez*jz) + ex*(5*(-25 + 4*e_per**2)*ey**2 + 25*(1 + 4*e_per**2)*ez**2 - 45*jx**2 - 76*e_per**2*jx**2 + 25*jy**2 - 4*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 3*(5*(-5 + 8*e_per**2)*ex**3 - 4*jx*(ey*(jy + 8*e_per**2*jy) + (1 - 4*e_per**2)*ez*jz) + ex*(-5*(-5 + 8*e_per**2)*ey**2 - 5*(1 + 8*e_per**2)*ez**2 + 9*jx**2 + 24*e_per**2*jx**2 - 5*jy**2 + 8*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f1**2*f3**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*(4*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2)) + 3*e_per*(5*ex*ez**2 + 2*ey*jx*jy - 2*ex*jy**2 - ez*jx*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per))*f1**2*f2*f3**2 - 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*((2*jx*((-7 + 4*e_per**2)*ey*jy + (2 - 5*e_per**2)*ez*jz) + ex*(-25*(-2 + e_per**2)*ey**2 + 5*ez**2 - 4*jy**2 + e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi**4 + (2*jx*(-5*(7 + 2*e_per**2)*ey*jy + (10 + 53*e_per**2)*ez*jz) + ex*(25*(10 + e_per**2)*ey**2 - 5*(-5 + 6*e_per**2)*ez**2 - 20*jy**2 + 11*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 6*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.arccos(-(1/e_per))**4)*f2**2*f3**2 - 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*((-5*(-5 + 2*e_per**2)*ex**3 + 4*jx*(ey*(jy - 2*e_per**2*jy) + ez*jz) + ex*(5*(-5 + 2*e_per**2)*ey**2 + (5 - 10*e_per**2)*ez**2 - 9*jx**2 + 10*e_per**2*jx**2 + 5*jy**2 - 2*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi**5 + (-5*(-25 + 4*e_per**2)*ex**3 + 4*jx*(5*ey*(jy + 4*e_per**2*jy) + (5 - 6*e_per**2)*ez*jz) + ex*(5*(-25 + 4*e_per**2)*ey**2 + 25*(1 + 4*e_per**2)*ez**2 - 45*jx**2 - 76*e_per**2*jx**2 + 25*jy**2 - 4*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 6*(5*(-5 + 8*e_per**2)*ex**3 - 4*jx*(ey*(jy + 8*e_per**2*jy) + (1 - 4*e_per**2)*ez*jz) + ex*(-5*(-5 + 8*e_per**2)*ey**2 - 5*(1 + 8*e_per**2)*ez**2 + 9*jx**2 + 24*e_per**2*jx**2 - 5*jy**2 + 8*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f2**2*f3**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*((-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2)) + 3*e_per*(5*ex*ez**2 + 2*ey*jx*jy - 2*ex*jy**2 - ez*jx*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per))*f1*f2**2*f3**2))/np.pi + (ex*(18*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(9*(5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*(-27*(25*(-2 + e_per**2)*ex**2*ey + 2*(7 - 3*e_per**2)*ex*jx*jy - 2*(2 + 3*e_per**2)*ez*jy*jz + ey*(5*(-1 + e_per**2)*ez**2 + (4 - 3*e_per**2)*jx**2 + 5*(5 - 3*e_per**2)*jz**2))*np.pi**4 + 3*(25*(10 + e_per**2)*ex**2*ey + 2*(-35 + 3*e_per**2)*ex*jx*jy + 2*(10 - 51*e_per**2)*ez*jy*jz + 5*ey*((5 + 7*e_per**2)*ez**2 - (4 + 3*e_per**2)*jx**2 - (25 + 9*e_per**2)*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 2*(50*(-1 + e_per**2)*ex**2*ey + 2*(7 - 10*e_per**2)*ex*jx*jy + 4*(-1 + 13*e_per**2)*ez*jy*jz + ey*(-5*(1 + 2*e_per**2)*ez**2 + 2*(2 + e_per**2)*jx**2 + 5*(5 - 2*e_per**2)*jz**2))*np.arccos(-(1/e_per))**4)*f1**2*f2**2 - 18*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(9*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(-27*(5*(-5 + 3*e_per**2)*ex**2*ey - 5*(-5 + 3*e_per**2)*ey**3 + 4*(1 + e_per**2)*ex*jx*jy - 4*(-1 + e_per**2)*ez*jy*jz + ey*(5*(1 + e_per**2)*ez**2 + (5 - 3*e_per**2)*jx**2 - 9*jy**2 - e_per**2*jy**2 - 25*jz**2 + 15*e_per**2*jz**2))*np.pi**5 + 3*(5*(25 + 9*e_per**2)*ex**2*ey - 5*(25 + 9*e_per**2)*ey**3 + 4*(-5 + 19*e_per**2)*ex*jx*jy - 4*(5 + 7*e_per**2)*ez*jy*jz + ey*(5*(-5 + 19*e_per**2)*ez**2 - (25 + 9*e_per**2)*jx**2 + 45*jy**2 - 67*e_per**2*jy**2 + 125*jz**2 + 45*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 2*(5*(-5 + 2*e_per**2)*ex**2*ey - 5*(-5 + 2*e_per**2)*ey**3 + 4*(1 - 10*e_per**2)*ex*jx*jy + 4*(1 + 2*e_per**2)*ez*jy*jz + ey*((5 - 50*e_per**2)*ez**2 + (5 - 2*e_per**2)*jx**2 - 9*jy**2 + 42*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f1**2*f2**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*(9*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-50*(-1 + e_per**2)*ex**2*ey + 2*(-7 + 10*e_per**2)*ex*jx*jy + 4*(1 - 13*e_per**2)*ez*jy*jz + ey*(5*(1 + 2*e_per**2)*ez**2 - 2*(2 + e_per**2)*jx**2 + 5*(-5 + 2*e_per**2)*jz**2)) + 3*e_per*(5*ey*ez**2 - 2*ey*jx**2 + 2*ex*jx*jy - ez*jy*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per))*f1**2*f2**2*f3 + 12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*(-8*(25*(-2 + e_per**2)*ex**2*ey + 2*(7 - 3*e_per**2)*ex*jx*jy - 2*(2 + 3*e_per**2)*ez*jy*jz + ey*(5*(-1 + e_per**2)*ez**2 + (4 - 3*e_per**2)*jx**2 + 5*(5 - 3*e_per**2)*jz**2))*np.pi**4 + 2*(25*(10 + e_per**2)*ex**2*ey + 2*(-35 + 3*e_per**2)*ex*jx*jy + 2*(10 - 51*e_per**2)*ez*jy*jz + 5*ey*((5 + 7*e_per**2)*ez**2 - (4 + 3*e_per**2)*jx**2 - (25 + 9*e_per**2)*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 3*(50*(-1 + e_per**2)*ex**2*ey + 2*(7 - 10*e_per**2)*ex*jx*jy + 4*(-1 + 13*e_per**2)*ez*jy*jz + ey*(-5*(1 + 2*e_per**2)*ez**2 + 2*(2 + e_per**2)*jx**2 + 5*(5 - 2*e_per**2)*jz**2))*np.arccos(-(1/e_per))**4)*f1**2*f3**2 - 12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(-8*(5*(-5 + 3*e_per**2)*ex**2*ey - 5*(-5 + 3*e_per**2)*ey**3 + 4*(1 + e_per**2)*ex*jx*jy - 4*(-1 + e_per**2)*ez*jy*jz + ey*(5*(1 + e_per**2)*ez**2 + (5 - 3*e_per**2)*jx**2 - 9*jy**2 - e_per**2*jy**2 - 25*jz**2 + 15*e_per**2*jz**2))*np.pi**5 + 2*(5*(25 + 9*e_per**2)*ex**2*ey - 5*(25 + 9*e_per**2)*ey**3 + 4*(-5 + 19*e_per**2)*ex*jx*jy - 4*(5 + 7*e_per**2)*ez*jy*jz + ey*(5*(-5 + 19*e_per**2)*ez**2 - (25 + 9*e_per**2)*jx**2 + 45*jy**2 - 67*e_per**2*jy**2 + 125*jz**2 + 45*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 3*(5*(-5 + 2*e_per**2)*ex**2*ey - 5*(-5 + 2*e_per**2)*ey**3 + 4*(1 - 10*e_per**2)*ex*jx*jy + 4*(1 + 2*e_per**2)*ez*jy*jz + ey*((5 - 50*e_per**2)*ez**2 + (5 - 2*e_per**2)*jx**2 - 9*jy**2 + 42*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f1**2*f3**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*(4*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-50*(-1 + e_per**2)*ex**2*ey + 2*(-7 + 10*e_per**2)*ex*jx*jy + 4*(1 - 13*e_per**2)*ez*jy*jz + ey*(5*(1 + 2*e_per**2)*ez**2 - 2*(2 + e_per**2)*jx**2 + 5*(-5 + 2*e_per**2)*jz**2)) + 3*e_per*(5*ey*ez**2 - 2*ey*jx**2 + 2*ex*jx*jy - ez*jy*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per))*f1**2*f2*f3**2 + 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*((-25*(-2 + e_per**2)*ex**2*ey + 2*(-7 + 3*e_per**2)*ex*jx*jy + 2*(2 + 3*e_per**2)*ez*jy*jz + ey*(-5*(-1 + e_per**2)*ez**2 + (-4 + 3*e_per**2)*jx**2 + 5*(-5 + 3*e_per**2)*jz**2))*np.pi**4 + (25*(10 + e_per**2)*ex**2*ey + 2*(-35 + 3*e_per**2)*ex*jx*jy + 2*(10 - 51*e_per**2)*ez*jy*jz + 5*ey*((5 + 7*e_per**2)*ez**2 - (4 + 3*e_per**2)*jx**2 - (25 + 9*e_per**2)*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 6*(50*(-1 + e_per**2)*ex**2*ey + 2*(7 - 10*e_per**2)*ex*jx*jy + 4*(-1 + 13*e_per**2)*ez*jy*jz + ey*(-5*(1 + 2*e_per**2)*ez**2 + 2*(2 + e_per**2)*jx**2 + 5*(5 - 2*e_per**2)*jz**2))*np.arccos(-(1/e_per))**4)*f2**2*f3**2 - 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*((-5*(-5 + 3*e_per**2)*ex**2*ey + 5*(-5 + 3*e_per**2)*ey**3 - 4*(1 + e_per**2)*ex*jx*jy + 4*(-1 + e_per**2)*ez*jy*jz + ey*(-5*(1 + e_per**2)*ez**2 + (-5 + 3*e_per**2)*jx**2 + 9*jy**2 + e_per**2*jy**2 + 25*jz**2 - 15*e_per**2*jz**2))*np.pi**5 + (5*(25 + 9*e_per**2)*ex**2*ey - 5*(25 + 9*e_per**2)*ey**3 + 4*(-5 + 19*e_per**2)*ex*jx*jy - 4*(5 + 7*e_per**2)*ez*jy*jz + ey*(5*(-5 + 19*e_per**2)*ez**2 - (25 + 9*e_per**2)*jx**2 + 45*jy**2 - 67*e_per**2*jy**2 + 125*jz**2 + 45*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 6*(5*(-5 + 2*e_per**2)*ex**2*ey - 5*(-5 + 2*e_per**2)*ey**3 + 4*(1 - 10*e_per**2)*ex*jx*jy + 4*(1 + 2*e_per**2)*ez*jy*jz + ey*((5 - 50*e_per**2)*ez**2 + (5 - 2*e_per**2)*jx**2 - 9*jy**2 + 42*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f2**2*f3**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*((-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-50*(-1 + e_per**2)*ex**2*ey + 2*(-7 + 10*e_per**2)*ex*jx*jy + 4*(1 - 13*e_per**2)*ez*jy*jz + ey*(5*(1 + 2*e_per**2)*ez**2 - 2*(2 + e_per**2)*jx**2 + 5*(-5 + 2*e_per**2)*jz**2)) + 3*e_per*(5*ey*ez**2 - 2*ey*jx**2 + 2*ex*jx*jy - ez*jy*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per))*f1*f2**2*f3**2))/np.pi - 12*np.sqrt(1 - e_per**(-2))*e_per**2*ez*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*np.arccos(-1.0/e_per)*(2*np.sqrt(1 - e_per**(-2))*((ey*jx - ex*jy)*jz + e_per**2*(5*ex*ey*ez + 3*ez*jx*jy - ey*jx*jz + ex*jy*jz)) + e_per*(6*ez*jx*jy + ey*jx*jz + ex*jy*jz)*np.arccos(-1.0/e_per))*f1*f2*f3*(f2*f3 + f1*(f2 + f3)) - e_per*(-5 + 2*e_per**2)*(10*ex*ey*ez - ey*jx*jz - ex*jy*jz)*np.pi**2*np.arccos(-(1/e_per))*np.arccos(-1.0/e_per)*f1*f2*f3*(f2*f3 + f1*(9*f2 + 4*f3)) - (-5 + 2*e_per**2)*np.pi**2*np.arccos(-1.0/e_per)*(2*np.sqrt(1 - e_per**(-2))*((ey*jx - ex*jy)*jz + e_per**2*(5*ex*ey*ez + 3*ez*jx*jy - ey*jx*jz + ex*jy*jz)) + e_per*(6*ez*jx*jy + ey*jx*jz + ex*jy*jz)*np.arccos(-1.0/e_per))*f1*f2*f3*(f2*f3 + f1*(9*f2 + 4*f3)) + 36*np.sqrt(1 - e_per**(-2))*(-5 + 8*e_per**2)*(4*(ey*jx - ex*jy)*jz + e_per**2*(5*ex*ey*ez + 5*ez*jx*jy - ey*jx*jz + 7*ex*jy*jz))*np.arccos(-(1/e_per))**9*(f2**2*f3**2 + f1**2*(f2**2 + f3**2)) - np.sqrt(1 - e_per**(-2))*(1320*(-(ey*jx*jz) + ex*jy*jz) + 16*e_per**4*(55*ex*ey*ez + 55*ez*jx*jy + 21*ey*jx*jz + 45*ex*jy*jz) - e_per**2*(1225*ex*ey*ez + 1225*ez*jx*jy - 1173*ey*jx*jz + 2643*ex*jy*jz))*np.pi**2*np.arccos(-(1/e_per))**7*(f2**2*f3**2 + f1**2*(9*f2**2 + 4*f3**2)) + 2*np.sqrt(1 - e_per**(-2))*(e_per**4*(85*ex*ey*ez + 85*ez*jx*jy + 111*ey*jx*jz - 9*ex*jy*jz) + 240*(-(ey*jx*jz) + ex*jy*jz) - e_per**2*(175*ex*ey*ez + 175*ez*jx*jy + 141*ey*jx*jz + 69*ex*jy*jz))*np.pi**4*np.arccos(-(1/e_per))**5*(f2**2*f3**2 + f1**2*(81*f2**2 + 16*f3**2)) + np.arccos(-(1/e_per))**3*(e_per*(-5 + 8*e_per**2)*(10*ex*ey*ez - ey*jx*jz - ex*jy*jz)*np.arccos(-1.0/e_per)*f1*f2*f3*(f2*f3 + f1*(f2 + f3)) - np.sqrt(1 - e_per**(-2))*(-5 + 2*e_per**2)*(24*(-(ey*jx) + ex*jy)*jz + e_per**2*(5*ex*ey*ez + 5*ez*jx*jy + 15*ey*jx*jz - 9*ex*jy*jz))*np.pi**6*(f2**2*f3**2 + f1**2*(729*f2**2 + 64*f3**2))))))/(2.*e_per**4*np.sqrt(ex**2 + ey**2 + ez**2)*f1**2*f2**2*f3**2)
             
            #Delta_e_SO = (-3*Pi*(4*ey*ez*jz*(2*jx - 3*jy*Pi) - 2*ez**2*(jx*jy + 9*jx**2*Pi + 9*jy**2*Pi) + 3*ey**2*(-2*jx*jy + 25*ez**2*Pi - 6*jx**2*Pi + jz**2*Pi) + ex**2*(-6*jx*jy + 3*(25*ez**2 - 6*jy**2 + jz**2)*Pi) + ex*(4*ez*jz*(2*jy - 3*jx*Pi) + ey*(-25*ez**2 + 6*jx**2 + 6*jy**2 - 5*jz**2 + 36*jx*jy*Pi)))*\[Epsilon]SA**2)/(8.*Sqrt(ex**2 + ey**2 + ez**2)
    

    return Delta_e_SO,Delta_i


def compute_TO_prediction_only(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz):
    
#    ArcCos = np.arccos
    Sqrt = np.sqrt
#    eper = e_per
    Pi = np.pi

    epsilon = 1.0e-10
    if e_per <= 1.0 + epsilon: ### parabolic limit
        
        Delta_e_TO = (3*Pi*(-1200*ex**4*jz*Pi - 1800*ey**4*jz*Pi + 96*ez**2*jx*jz*(49*jy + 20*jx*Pi) - 6*ey**2*jz*(80*jx**2*Pi + 20*(80*ez**2 - 7*jy**2)*Pi + jx*jy*(371 - 192*Pi**2)) + ey*ez*(87*jx**3 + 285*jx*jy**2 + 720*jx**2*jy*Pi + 120*jy*(9*jy**2 + 16*jz**2)*Pi - 360*ez**2*jx*(21 + 16*Pi**2) - 8*jx*jz**2*(7 + 48*Pi**2)) - 15*ey**3*ez*(520*jy*Pi + jx*(-387 + 384*Pi**2)) + 15*ex**3*(466*ey*jz + 80*ez*jx*Pi + ez*jy*(369 + 384*Pi**2)) - 3*ex**2*(1000*ey**2*jz*Pi + 15*ey*ez*(160*jy*Pi + jx*(-159 + 128*Pi**2)) + 2*jz*(120*jx**2*Pi + 20*(160*ez**2 + jy**2)*Pi + jx*jy*(581 + 192*Pi**2))) + ex*(9390*ey**3*jz + 4*ey*jz*(840*ez**2 + 180*jx*jy*Pi + 8*jx**2*(-7 + 36*Pi**2) - 3*jy**2*(217 + 96*Pi**2)) + 15*ey**2*ez*(40*jx*Pi + jy*(811 + 384*Pi**2)) + ez*(-313*jx**2*jy - 927*jy**3 - 616*jy*jz**2 + 720*jx**3*Pi + 120*jx*(9*jy**2 + 32*jz**2)*Pi + 384*jy*jz**2*Pi**2 - 120*ez**2*(80*jx*Pi + jy*(133 - 48*Pi**2)))))*eps_SA**3)/(512.*Sqrt(ex**2 + ey**2 + ez**2))

        if args.include_octupole_terms==True:
            Delta_e_TO += eps_oct*(-3*Pi*(-1593720*ex**5*jz*Pi + 3*ey**5*jz*(1839283 + 7680*Pi**2) + 48*ey**4*ez*(45290*jx*Pi - jy*(58651 + 15360*Pi**2)) + 2*ey**3*jz*(488480 - 4085587*jy**2 - 3027816*jz**2 - 2041920*jx*jy*Pi + 15360*Pi**2 + 23040*jy**2*Pi**2 + 34560*jz**2*Pi**2 + jx**2*(1731827 - 464640*Pi**2) - 8*ez**2*(138617 + 62880*Pi**2)) + 3*ex**4*(898880*ez*jx*Pi + ey*jz*(2412271 + 7680*Pi**2) + 48*ez*jy*(26069 + 67360*Pi**2)) + ey*jz*(-260315*jx**4 - 74880*jx**3*jy*Pi - 960*jx*jy*(12 + 121*jy**2 - 406*jz**2)*Pi + 10240*ez**4*(-161 + 192*Pi**2) + jy**4*(495781 + 115200*Pi**2) - 144*jy**2*(128*(11 + 5*Pi**2) + jz**2*(-1859 + 800*Pi**2)) + 2*jx**2*(101376 - 8*jz**2*(15191 + 2400*Pi**2) + 3*jy**2*(48587 + 19200*Pi**2)) + 16*ez**2*(-387480*jx*jy*Pi + jy**2*(24387 - 99360*Pi**2) + 80*(161 - 192*Pi**2) + jx**2*(502433 + 28320*Pi**2))) + 8*ey**2*ez*(56940*jx*jy**2*Pi + 60*jx*(520 - 34316*ez**2 + 5563*jx**2 + 9248*jz**2)*Pi + jy**3*(843447 - 115200*Pi**2) + jy*(120*(-431 + 576*Pi**2) + 18*jz**2*(67287 + 5920*Pi**2) - 5*jx**2*(-76163 + 31872*Pi**2) + 6*ez**2*(-514567 + 90720*Pi**2))) - 4*ex**3*(959940*ey**2*jz*Pi + 24*ey*ez*(96175*jy*Pi + 6*jx*(-14443 + 18120*Pi**2)) + jz*(250620*jx**2*Pi + 60*(940 + 162316*ez**2 - 43087*jy**2 - 5124*jz**2)*Pi + jx*jy*(774103 + 720000*Pi**2))) + 16*ez*(240*ez**4*(460*jx*Pi + jy*(1687 - 384*Pi**2)) + jz**2*(120*jx*(155 - 422*jx**2)*Pi - 45480*jx*jy**2*Pi + jy**3*(5863 - 2400*Pi**2) + jy*(770 + 3840*Pi**2 + jx**2*(13981 - 2400*Pi**2))) - 5*ez**2*(-56640*jx*jy**2*Pi + 120*jx*(23 - 375*jx**2 + 248*jz**2)*Pi + 9*jy**3*(-9251 + 2144*Pi**2) + jy*(10122 - 2304*Pi**2 + 16*jz**2*(77 + 384*Pi**2) + 3*jx**2*(-11647 + 6432*Pi**2)))) + 2*ex**2*(-8*ey*ez**2*jz*(59729 + 62880*Pi**2) - 8*ez**3*(2315520*jx*Pi + jy*(2282149 - 682080*Pi**2)) + ey*jz*(410000 - 7449407*jy**2 - 1611432*jz**2 - 7710720*jx*jy*Pi + 15360*Pi**2 - 1904640*jy**2*Pi**2 + 34560*jz**2*Pi**2 + 3*ey**2*(2428777 + 7680*Pi**2) + 9*jx**2*(-358217 + 162560*Pi**2)) + 4*ez*(617400*jx*jy**2*Pi + 60*jx*(9259*ey**2 + 4*(325 + 31*jx**2 + 6383*jz**2))*Pi + jy**3*(301009 - 288960*Pi**2) + jy*(37860 - 614906*jz**2 + 80640*Pi**2 + 143040*jz**2*Pi**2 + 36*ey**2*(49547 + 31120*Pi**2) - 3*jx**2*(-36439 + 81600*Pi**2)))) - 4*ex*(561510*ey**4*jz*Pi + ey**2*jz*(-1802580*jx**2*Pi + 60*(2420 + 93612*ez**2 - 30067*jy**2 - 3108*jz**2)*Pi + jx*jy*(1793083 - 1207680*Pi**2)) + 120*ey**3*ez*(20123*jy*Pi + jx*(-1199 + 21744*Pi**2)) + 4*ey*ez*(235230*jx**2*jy*Pi + 30*jy*(940 + 5192*ez**2 - 6939*jy**2 - 4656*jz**2)*Pi + jx**3*(143319 - 64800*Pi**2) + jx*(-3630 - 925654*jz**2 + 5760*Pi**2 + 18240*jz**2*Pi**2 + jy**2*(117649 - 108960*Pi**2) + ez**2*(678458 + 409920*Pi**2))) + jz*(-330*jx**4*Pi - 60*jx**2*(-316 + 61076*ez**2 - 219*jy**2 + 1428*jz**2)*Pi - 30*(138240*ez**4 + 8*ez**2*(-2160 + 16591*jy**2) - 21*jy**2*(8 + 5*jy**2 - 24*jz**2))*Pi - jx**3*jy*(188353 + 28800*Pi**2) + jx*jy*(-(jy**2*(182039 + 28800*Pi**2)) + 8*ez**2*(40207 + 63840*Pi**2) + 8*(9273 + 2880*Pi**2 + jz**2*(-3289 + 2400*Pi**2))))))*eps_SA**3)/(65536.*Sqrt(ex**2 + ey**2 + ez**2))
            Delta_e_TO += eps_oct**2*(-15*Pi*(148440600*ex**6*jz*Pi + 103458600*ey**6*jz*Pi + ey**5*(426531000*ez*jy*Pi - 630*ex*jz*(1600829 + 3840*Pi**2) + ez*jx*(58881979 + 11289600*Pi**2)) + 12*ex**4*jz*(1866060*jx**2*Pi + 140*(22512 + 2332842*ez**2 - 830105*jy**2 - 130410*jz**2)*Pi + jx*jy*(101496629 + 29102080*Pi**2)) - ex**5*ez*(38565240*jx*Pi + jy*(39314977 + 923328000*Pi**2)) + 64*ez**2*jz*(4418190*jx**4*Pi + 420*jx**2*(-4970 + 39760*ez**2 + 22871*jy**2)*Pi + 210*(560 + 35840*ez**4 - 13020*jy**2 + 19807*jy**4 + 1120*ez**2*(-8 + 93*jy**2))*Pi - 9*jx**3*jy*(754633 + 112000*Pi**2) - 5*jx*jy*(-541772 + 5376*Pi**2 - 224*ez**2*(-19349 + 192*Pi**2) + 3*jy**2*(284141 + 67200*Pi**2))) + 4*ex**2*jz*(-5210730*jx**4*Pi - 420*jx**2*(2127506*ez**2 - 364255*jy**2 - 6*(3036 + 79*jz**2))*Pi - 210*(6578432*ez**4 - 487539*jy**4 + 224*(-5 + 201*jz**2) - 4*jy**2*(-83420 + 5643*jz**2) + 4*ez**2*(359151*jy**2 - 56*(3631 + 1608*jz**2)))*Pi + jx**3*jy*(31045417 - 15052800*Pi**2) + jx*jy*(-39283816 - 43853238*jz**2 + 9891840*Pi**2 + 6182400*jz**2*Pi**2 - 75*jy**2*(-171301 + 376320*Pi**2) + 2*ez**2*(382511327 + 85397760*Pi**2))) + ey**4*(397693800*ex**2*jz*Pi - 4*jz*(52486140*jx**2*Pi + 420*(-21000 + 108534*ez**2 + 74689*jy**2 + 104202*jz**2)*Pi + jx*jy*(175372661 - 9192960*Pi**2)) + ex*ez*(-258776280*jx*Pi + jy*(-93213781 + 172408320*Pi**2))) - 2*ex**3*ez*(481139400*jx*jy**2*Pi - 840*jx*(-91720 + 3274102*ez**2 + 38539*jx**2 - 899118*jz**2)*Pi - 5*jy**3*(-48011749 + 52405248*Pi**2) + jy*(-40012456 + 309696222*jz**2 + 70318080*Pi**2 + 72737280*jz**2*Pi**2 - 3*jx**2*(-44741551 + 55462400*Pi**2) + 2*ez**2*(-935900339 + 369196800*Pi**2))) - ex*ez*(-10742760*jx*jy**4*Pi + 1680*jx*jy**2*(-6304 + 1107246*ez**2 + 8629*jx**2 - 337094*jz**2)*Pi + 840*jx*(2189824*ez**4 + 29055*jx**4 + 224*(17 + 790*jz**2) - 4*jx**2*(7256 + 126975*jz**2) + 28*ez**2*(56133*jx**2 - 16*(679 + 3160*jz**2)))*Pi + 3*jy**5*(-17195893 + 4390400*Pi**2) + jy**3*(66417472 + 597951068*jz**2 - 13762560*Pi**2 - 20966400*jz**2*Pi**2 - 28*ez**2*(-26801203 + 8712960*Pi**2) + jx**2*(-78005818 + 33331200*Pi**2)) + jy*(3584*ez**4*(153043 + 4800*Pi**2) + 5*jx**4*(-804991 + 4032000*Pi**2) + 448*(6138 + 5760*Pi**2 + 5*jz**2*(-92741 + 9408*Pi**2)) - 28*ez**2*(3234352 + 814080*Pi**2 + 640*jz**2*(-92741 + 9408*Pi**2) + 5*jx**2*(2579621 + 1837824*Pi**2)) + 4*jx**2*(96*(53801 - 50400*Pi**2) + jz**2*(156820763 + 7123200*Pi**2)))) + 2*ey**3*(-3150*ex**3*jz*(322249 + 768*Pi**2) + 40*ex*ez**2*jz*(-26135527 + 3999744*Pi**2) + 6*ez**3*(197459080*jy*Pi + 11*jx*(-14088139 + 224000*Pi**2)) + 8*ex*jz*(42652680*jx*jy*Pi + jx**2*(-47218478 + 7156800*Pi**2) - 7*(266046 + 34560*Pi**2 + 9*jz**2*(-1518517 + 7680*Pi**2) + jy**2*(-14634161 + 730560*Pi**2))) + ez*(-60707640*jx**2*jy*Pi + 840*jy*(-92112 + 982965*ex**2 + 22565*jy**2 + 153234*jz**2)*Pi + jx**3*(162513403 - 33062400*Pi**2) + jx*(-13747448 + 717339150*jz**2 + 7526400*Pi**2 - 20805120*jz**2*Pi**2 - 5*jy**2*(-27464243 + 6623232*Pi**2) + 15*ex**2*(3294239 + 37277184*Pi**2)))) + 2*ey**2*(221337900*ex**4*jz*Pi - 84*ex**2*jz*(6708460*jx**2*Pi - 20*(33012 + 691618*ez**2 - 315529*jy**2 - 106722*jz**2)*Pi + jx*jy*(-172001 + 3759360*Pi**2)) - 3*ex**3*ez*(45735480*jx*Pi + jy*(222906361 + 125153280*Pi**2)) - ex*ez*(-387276120*jx*jy**2*Pi + 840*jx*(88392 - 2512702*ez**2 + 217873*jx**2 + 454102*jz**2)*Pi + jy**3*(631538437 - 85532160*Pi**2) + jy*(-153259656 + 58060800*Pi**2 + jx**2*(539531305 - 181063680*Pi**2) + 2*jz**2*(884960131 + 51098880*Pi**2) + 10*ez**2*(-100236235 + 57732864*Pi**2))) + 2*jz*(76978650*jx**4*Pi - 420*jx**2*(83344 + 377598*ez**2 - 96841*jy**2 - 35702*jz**2)*Pi - 210*(1754368*ez**4 - 121445*jy**4 + 672*(-15 + 43*jz**2) - 4*ez**2*(34664 + 42023*jy**2 + 57792*jz**2) + jy**2*(85984 + 70284*jz**2))*Pi + jx**3*jy*(87173563 - 4300800*Pi**2) + jx*jy*(-21179048 - 39542074*jz**2 - 6021120*Pi**2 + 7795200*jz**2*Pi**2 + jy**2*(25494157 + 8870400*Pi**2) + 2*ez**2*(-244959143 + 60076800*Pi**2)))) + ey*(-86016*ex*ez**4*jz*(9097 + 6480*Pi**2) + 3584*ez**5*(-146160*jy*Pi + jx*(205777 + 33600*Pi**2)) + 8*ex*ez**2*jz*(14986048 + 285718389*jy**2 + 111707904*jz**2 - 124882800*jx*jy*Pi + 9569280*Pi**2 + 2607360*jy**2*Pi**2 + 2580480*jz**2*Pi**2 - jx**2*(436035941 + 22713600*Pi**2) + ex**2*(-245490146 + 39997440*Pi**2)) + 4*ez**3*(-121369080*jx**2*jy*Pi + 840*jy*(16352 + 534063*ex**2 - 329539*jy**2 - 98560*jz**2)*Pi + 21*jx**3*(1147289 + 172800*Pi**2) + ex**2*jx*(-323839087 + 87924480*Pi**2) + 7*jx*(-4280848 - 537600*Pi**2 + 17280*jz**2*(4037 + 64*Pi**2) + 5*jy**2*(6452083 + 198912*Pi**2))) - 2*ex*jz*(1379840 - 51718656*jy**2 + 133713177*jy**4 + 55853952*jz**2 - 122669404*jy**2*jz**2 + 407712480*jx**3*jy*Pi + 3360*jx*jy*(-55508 + 51329*jy**2 + 30826*jz**2)*Pi + 430080*Pi**2 - 30750720*jy**2*Pi**2 + 40454400*jy**4*Pi**2 + 1290240*jz**2*Pi**2 - 12364800*jy**2*jz**2*Pi**2 + 315*ex**4*(1508429 + 3840*Pi**2) - 15*jx**4*(-7864847 + 492800*Pi**2) + 8*ex**2*(-130111800*jx*jy*Pi + 70*jx**2*(260945 + 352032*Pi**2) - 3*jy**2*(54954937 + 8895040*Pi**2) + 21*(6094 + 11520*Pi**2 + 9*jz**2*(-428637 + 2560*Pi**2))) - 2*jx**2*(35*jy**2*(-4860091 + 280320*Pi**2) + 6*(6330280 - 89600*Pi**2 + 29*jz**2*(622127 + 44800*Pi**2)))) + ez*(-3705845*jx**5 - 53219880*jx**4*jy*Pi - 1680*jx**2*jy*(-22040 - 612285*ex**2 + 71027*jy**2 + 119642*jz**2)*Pi + 840*jy*(1430859*ex**4 - 77705*jy**4 - 2*ex**2*(12512 + 583119*jy**2 - 87754*jz**2) + 224*(7 + 220*jz**2) + 4*jy**2*(16860 + 4223*jz**2))*Pi + jx**3*(8259680 + 822064188*jz**2 + 19891200*jz**2*Pi**2 + jy**2*(42784082 - 6988800*Pi**2) - 6*ex**2*(-60084973 + 37990400*Pi**2)) + jx*(jy**4*(80969031 - 6988800*Pi**2) + 13*ex**4*(-68609789 + 85155840*Pi**2) - 12096*(-286 + 5*jz**2*(4037 + 64*Pi**2)) + 20*jy**2*(jz**2*(36844951 - 1478400*Pi**2) + 8*(47795 + 34944*Pi**2)) - 2*ex**2*(495352 - 19783680*Pi**2 + 22*jz**2*(35728079 + 2284800*Pi**2) + jy**2*(-236617211 + 305195520*Pi**2))))))*eps_SA**3)/(7.340032e6*Sqrt(ex**2 + ey**2 + ez**2))
            Delta_e_TO += eps_oct**3*(-75*Pi*(-169317540*ex**7*jz*Pi - 315*ey**7*jz*(-307531 + 2304*Pi**2) + 210*ey**6*ez*(1137240*jx*Pi + jy*(7510135 + 89856*Pi**2)) + 3*ex**6*(-178262280*ez*jx*Pi + 63*ey*jz*(6133391 + 11520*Pi**2) + ez*jy*(191774707 + 335543040*Pi**2)) + ey**5*jz*(507694320*jx*jy*Pi - 1527*jy**2*(778287 + 8960*Pi**2) + 9*ez**2*(-211157999 + 1706880*Pi**2) + 4*jx**2*(-236111293 + 4233600*Pi**2) + 63*(3089177 - 30720*Pi**2 + 5*jz**2*(-312143 + 1152*Pi**2))) - 4*ey**4*ez*(149115960*jx*jy**2*Pi + 2520*jx*(-4587 - 53541*ez**2 + 8902*jx**2 + 39459*jz**2)*Pi + 7*jy**3*(-511801 + 2300160*Pi**2) + jy*(63439129 + 135129124*jz**2 - 4515840*Pi**2 - 211680*jz**2*Pi**2 + jx**2*(307354014 - 6720000*Pi**2) + 2*ez**2*(-605792837 + 15780240*Pi**2))) - ey**3*jz*(60792732 - 367924570*jy**2 + 92615433*jy**4 + 293441148*jz**2 - 357218082*jy**2*jz**2 + 360501120*jx**3*jy*Pi - 10080*jx*jy*(2113*jy**2 - 3*(437 + 4762*jz**2))*Pi + 1290240*Pi**2 - 17633280*jy**2*Pi**2 + 6585600*jy**4*Pi**2 - 967680*jz**2*Pi**2 + 7929600*jy**2*jz**2*Pi**2 + 32*ez**4*(287186471 + 2499840*Pi**2) + 5*jx**4*(-45537497 + 5241600*Pi**2) + jx**2*(-22014694 - 22579200*Pi**2 + 360*jy**2*(86939 + 147840*Pi**2) + jz**2*(-471607058 + 23788800*Pi**2)) + 2*ez**2*(-558905760*jx*jy*Pi + 1008*jz**2*(-1164449 + 3840*Pi**2) - 10*(81754387 + 1016064*Pi**2) + jx**2*(3096800421 + 36086400*Pi**2) + jy**2*(1262463917 + 104603520*Pi**2))) - ex**5*(557485740*ey**2*jz*Pi + 5*jz*(-70351092*jx**2*Pi + 252*(65394 + 2805498*ez**2 - 1355915*jy**2 - 252882*jz**2)*Pi + jx*jy*(555709919 + 88784640*Pi**2)) + 3*ey*ez*(877779000*jy*Pi + jx*(-603578251 + 466233600*Pi**2))) + ey*jz*(-205661070*jx**6 + 12345476*jy**2 - 56047849*jy**4 + 33735225*jy**6 + 62078016*jz**2 - 244802844*jy**2*jz**2 + 198823755*jy**4*jz**2 - 226422000*jx**5*jy*Pi - 10080*jx**3*jy*(-24613 + 35520*jy**2 + 6540*jz**2)*Pi - 5040*jx*jy*(13348 + 24135*jy**4 - 4044*jz**2 + jy**2*(-36998 + 7140*jz**2))*Pi - 4730880*jy**2*Pi**2 + 11827200*jy**4*Pi**2 - 7392000*jy**6*Pi**2 + 645120*jz**2*Pi**2 - 1612800*jy**2*jz**2*Pi**2 + 1008000*jy**4*jz**2*Pi**2 - 28672*ez**6*(95953 + 480*Pi**2) + 15*jx**4*(16781193 + jz**2*(35117329 + 604800*Pi**2) + jy**2*(-14692141 + 1478400*Pi**2)) - 32*ez**4*(140540400*jx*jy*Pi + 325*jx**2*(632005 + 16128*Pi**2) + 5*jy**2*(-54501733 + 2456832*Pi**2) - 224*(95953 + 480*Pi**2 + 12*jz**2*(46189 + 480*Pi**2))) + 2*jx**2*(30*jy**4*(-550343 + 246400*Pi**2) + 25*jy**2*(2639923 - 236544*Pi**2 + 9*jz**2*(1362569 + 40320*Pi**2)) - 6*(5652075 + jz**2*(31331443 + 403200*Pi**2))) + ez**2*(-2054747520*jx**3*jy*Pi - 20160*jx*jy*(-54581 + 103371*jy**2 + 8088*jz**2)*Pi + jy**4*(2563000297 - 108393600*Pi**2) - 5*jx**4*(581712703 + 15120000*Pi**2) - 448*(95953 + 480*Pi**2 + 48*jz**2*(46189 + 480*Pi**2)) + 12*jy**2*(-99066539 + 7248640*Pi**2 + 24*jz**2*(6800079 + 44800*Pi**2)) + 2*jx**2*(650*(1049389 + 16128*Pi**2) + 48*jz**2*(31331443 + 403200*Pi**2) + 35*jy**2*(9991553 + 1265280*Pi**2)))) - 2*ey**2*ez*(-17723160*jx*jy**4*Pi - 5040*jx*jy**2*(1041 - 32448*ez**2 + 6112*jx**2 + 9328*jz**2)*Pi - 2520*jx*(92 + 212864*ez**4 + 20935*jx**4 + 64616*jz**2 + 4*ez**2*(-6836 + 43843*jx**2 - 129232*jz**2) - 2*jx**2*(6531 + 110422*jz**2))*Pi + jy**5*(-32258959 + 7123200*Pi**2) + 2*jy**3*(18604729 + 327764073*jz**2 - 5806080*Pi**2 - 4334400*jz**2*Pi**2 + jx**2*(43269836 + 9945600*Pi**2) + ez**2*(648205987 + 49237440*Pi**2)) + jy*(467324 - 334332856*jz**2 + 4730880*Pi**2 - 1128960*jz**2*Pi**2 + jx**4*(108772535 - 672000*Pi**2) + 384*ez**4*(8346481 + 959840*Pi**2) + 2*ez**2*(32*jz**2*(41791607 + 141120*Pi**2) - 40*(5054621 + 1048992*Pi**2) + 33*jx**2*(19514417 + 1131200*Pi**2)) - 2*jx**2*(32012123 + 2688000*Pi**2 + jz**2*(-173782843 + 14952000*Pi**2)))) - 4*ez*(1136790*jx**6*jy - 4749576*jy**5 + 4164150*jy**7 + 15775760*jy*jz**2 - 72298226*jy**3*jz**2 + 72144345*jy**5*jz**2 + 37800*jx**5*(88*jy**2 - 749*jz**2)*Pi + 5040*jx**3*(1680*jy**4 + 4123*jz**2 - 33*jy**2*(16 + 455*jz**2))*Pi + 2520*jx*(2040*jy**6 - 1400*jz**2 + 10766*jy**2*jz**2 - 3*jy**4*(544 + 5425*jz**2))*Pi - 1720320*jy**3*Pi**2 + 4300800*jy**5*Pi**2 - 2688000*jy**7*Pi**2 - 376320*jy*jz**2*Pi**2 + 940800*jy**3*jz**2*Pi**2 - 588000*jy**5*jz**2*Pi**2 + 7168*ez**6*(39060*jx*Pi + jy*(-107393 + 23520*Pi**2)) - 15*jx**4*jy*(132600 - 7*jz**2*(1022177 + 74400*Pi**2) + 2*jy**2*(-56599 + 89600*Pi**2)) + jx**2*(jy**5*(4443090 - 5376000*Pi**2) - 14*jy*jz**2*(6270979 + 48000*Pi**2) + 6*jy**3*(35*jz**2*(788489 + 34400*Pi**2) + 8*(-88439 + 89600*Pi**2))) - 16*ez**4*(-8429400*jx*jy**2*Pi - 2520*jx*(4317*jx**2 - 56*(31 + 100*jz**2))*Pi + 3*jy**3*(9995557 + 1196160*Pi**2) + jy*(jx**2*(32658483 - 6249600*Pi**2) + 112*(-107393 + 23520*Pi**2 + 20*jz**2*(-28171 + 672*Pi**2)))) + ez**2*(46320120*jx*jy**4*Pi + 5040*jx*jy**2*(-3345 + 11049*jx**2 - 43064*jz**2)*Pi + 2520*jx*(7965*jx**4 + 56*(31 + 400*jz**2) - 2*jx**2*(4317 + 32984*jz**2))*Pi - 21*jy**5*(445627 + 1442400*Pi**2) + jy**3*(-784*jz**2*(-737737 + 9600*Pi**2) - 6*jx**2*(10071269 + 3410400*Pi**2) + 6*(9995557 + 3489920*Pi**2)) + jy*(1755*jx**4*(-22149 + 5600*Pi**2) + 112*(-107393 + 23520*Pi**2 + 80*jz**2*(-28171 + 672*Pi**2)) + 2*jx**2*(32658483 - 6249600*Pi**2 + 56*jz**2*(6270979 + 48000*Pi**2))))) + ex**4*(-9*ey*ez**2*jz*(-78496301 + 89308800*Pi**2) - 12*ez**3*(640849440*jx*Pi + jy*(122388071 - 218237600*Pi**2)) + ey*jz*(193415859 - 4955309033*jy**2 - 2039371425*jz**2 - 2082099600*jx*jy*Pi + 3870720*Pi**2 - 481393920*jy**2*Pi**2 + 7620480*jz**2*Pi**2 + 63*ey**2*(40069541 + 57600*Pi**2) + 56*jx**2*(8867771 + 10756800*Pi**2)) + 2*ez*(1422043560*jx*jy**2*Pi - 2520*jx*(-57470 + 28806*ey**2 + 5027*jx**2 - 232964*jz**2)*Pi - 5*jy**3*(-86460679 + 92322048*Pi**2) + jy*(-25*jx**2*(-29854577 + 9757440*Pi**2) + 18*ey**2*(123540639 + 18771200*Pi**2) + 6*(-41823795 + 22256640*Pi**2 + jz**2*(98427381 + 15705760*Pi**2))))) - 2*ex**3*(289221030*ey**4*jz*Pi + ey**2*jz*(-1013197500*jx**2*Pi + 3780*(29778 + 142386*ez**2 - 193247*jy**2 - 84042*jz**2)*Pi + jx*jy*(273648269 - 427902720*Pi**2)) + 3*ey**3*ez*(710143560*jy*Pi + jx*(-245503669 + 240172800*Pi**2)) + jz*(2220750*jx**4*Pi - 1260*jx**2*(-28582 + 2839042*ez**2 - 785031*jy**2 + 8214*jz**2)*Pi - 630*(11002688*ez**4 - 766313*jy**4 + 4*ez**2*(-338794 + 197685*jy**2 - 236880*jz**2) + jy**2*(528268 - 102628*jz**2) + 2520*(-1 + 47*jz**2))*Pi - 5*jx**3*jy*(81853697 + 12499200*Pi**2) + jx*jy*(112*ez**2*(9072349 + 8997360*Pi**2) - jy**2*(414126581 + 141523200*Pi**2) + 6*(-4271007 + 8780800*Pi**2 + 336*jz**2*(-156611 + 11800*Pi**2)))) + ey*ez*(1538613720*jx**2*jy*Pi + 2520*jy*(-19616 + 1073522*ez**2 - 646079*jy**2 + 56566*jz**2)*Pi + jx**3*(760170999 - 221625600*Pi**2) + jx*(ez**2*(-2314207142 + 65990400*Pi**2) - 7*jy**2*(-74503697 + 92240640*Pi**2) - 10*(3239548 - 5268480*Pi**2 + jz**2*(97613269 + 24807552*Pi**2))))) + ex*(-190273860*ey**6*jz*Pi + 15*ey**4*jz*(58902060*jx**2*Pi + 84*(-89082 + 1013742*ez**2 + 138247*jy**2 + 269514*jz**2)*Pi + jx*jy*(91271335 - 5417216*Pi**2)) - 21*ey**5*ez*(76291560*jy*Pi + jx*(-25486441 + 2016000*Pi**2)) - 2*ey**2*jz*(572927670*jx**4*Pi - 1260*jx**2*(173474 + 1062458*ez**2 + 143061*jy**2 - 91874*jz**2)*Pi - 630*(4896320*ez**4 - 224981*jy**4 + 4*ez**2*(-106306 + 48481*jy**2 - 224784*jz**2) + 168*(-139 + 669*jz**2) + 68*jy**2*(1679 + 2091*jz**2))*Pi + jx**3*jy*(1147307479 - 19488000*Pi**2) + jx*jy*(-491358114 + 15569216*jz**2 - 22794240*Pi**2 + 55507200*jz**2*Pi**2 + 336*ez**2*(-9182173 + 2333200*Pi**2) + jy**2*(735596167 + 39110400*Pi**2))) + 2*ey**3*ez*(174784680*jx**2*jy*Pi + 2520*jy*(71840 - 1580186*ez**2 + 119199*jy**2 - 105646*jz**2)*Pi + jx**3*(-1286151071 + 115180800*Pi**2) + jx*(149296888 - 29030400*Pi**2 + ez**2*(2186414126 - 46368000*Pi**2) + jy**2*(417689809 + 35185920*Pi**2) + 2*jz**2*(-576007627 + 61165440*Pi**2))) + jz*(-11434500*jx**6*Pi - 1260*jx**4*(1318298*ez**2 - 5*(2134 + 35727*jy**2 + 14634*jz**2))*Pi - 1260*(3555328*ez**6 + 64*ez**4*(35501*jy**2 - 448*(31 + 6*jz**2)) - 3*(-4 + 5*jy**2)*(7359*jy**4 - 224*jz**2 - 2*jy**2*(2257 + 253*jz**2)) + ez**2*(-224638*jy**4 + 1792*(31 + 24*jz**2) + 8*jy**2*(18667 + 2712*jz**2)))*Pi - 1260*jx**2*(5326016*ez**4 - 306015*jy**4 + jy**2*(195412 - 61980*jz**2) + 8*(275 + 3699*jz**2) + 4*ez**2*(316723*jy**2 - 18*(9491 + 3288*jz**2)))*Pi - 15*jx**5*jy*(-14540527 + 672000*Pi**2) - 10*jx**3*jy*(29211754 - 1075200*Pi**2 - 48*jz**2*(-372359 + 25200*Pi**2) - 528*ez**2*(1184391 + 57680*Pi**2) + 21*jy**2*(-1305487 + 236800*Pi**2)) + jx*jy*(-2496*ez**4*(-6627841 + 8960*Pi**2) - 105*jy**4*(-1028653 + 377600*Pi**2) + 4*jy**2*(-57275647 + 8601600*Pi**2 + 210*jz**2*(-152711 + 4800*Pi**2)) - 8*(-9598303 + 268800*Pi**2 + jz**2*(-7675239 + 403200*Pi**2)) + 8*ez**2*(31*(-10815233 + 80640*Pi**2) + 24*jz**2*(-2558413 + 134400*Pi**2) + jy**2*(606924747 + 3998400*Pi**2)))) - ey*ez*(-62881560*jx**4*jy*Pi - 5040*jx**2*jy*(-10060 + 2766*ez**2 + 97343*jy**2 + 434490*jz**2)*Pi - 2520*jy*(9552 + 851456*ez**4 + 93365*jy**4 - 159072*jz**2 + 4*jy**2*(-27734 + 23069*jz**2) + 4*ez**2*(-45712 + 464023*jy**2 + 318144*jz**2))*Pi + 35*jx**5*(3778081 + 288000*Pi**2) + 2*jx**3*(-31680220 + 704352766*jz**2 - 5376000*Pi**2 + 83462400*jz**2*Pi**2 + 26*ez**2*(65462881 + 67200*Pi**2) + 7*jy**2*(-11788249 + 9849600*Pi**2)) + jx*(512*ez**4*(15881437 + 420000*Pi**2) + jy**4*(-327750193 + 74054400*Pi**2) + 48*(158587 + 44800*Pi**2 - 2*jz**2*(6463171 + 351680*Pi**2)) - 4*jy**2*(26*(-2660669 + 698880*Pi**2) + jz**2*(-740539839 + 34339200*Pi**2)) + 4*ez**2*(3*jy**2*(398617791 + 64247680*Pi**2) + 16*(-16832959 - 688800*Pi**2 + 12*jz**2*(6463171 + 351680*Pi**2)))))) + ex**2*(32*ey*ez**4*jz*(-215452523 + 43841280*Pi**2) + 256*ez**5*(19270440*jx*Pi + jy*(5139563 + 769440*Pi**2)) - 2*ey*ez**2*jz*(-579325318 + 3036777601*jy**2 - 736143408*jz**2 - 1981959840*jx*jy*Pi + 92843520*Pi**2 - 7271040*jy**2*Pi**2 + 19353600*jz**2*Pi**2 + jx**2*(212107753 - 299537280*Pi**2) + 27*ey**2*(16996083 + 14600320*Pi**2)) + 4*ez**3*(1564065720*jx*jy**2*Pi - 2520*jx*(74696 + 790987*ey**2 - 287947*jx**2 + 396512*jz**2)*Pi + 4*jy**3*(77085151 + 8337840*Pi**2) + jy*(-42*jx**2*(1102101 + 3490720*Pi**2) + ey**2*(1584961729 + 613341120*Pi**2) - 8*(5116969 + 3672480*Pi**2 + 260*jz**2*(122617 + 47040*Pi**2)))) + ey*jz*(-37105068 - 372039278*jy**2 + 1494159605*jy**4 - 184035852*jz**2 + 394855370*jy**2*jz**2 + 2769883200*jx**3*jy*Pi + 10080*jx*jy*(-102147 + 35075*jy**2 + 50770*jz**2)*Pi + 1290240*Pi**2 - 132249600*jy**2*Pi**2 + 176736000*jy**4*Pi**2 + 4838400*jz**2*Pi**2 - 39648000*jy**2*jz**2*Pi**2 + 63*ey**4*(22713103 + 11520*Pi**2) - 125*jx**4*(5486401 + 532224*Pi**2) - 10*jx**2*(-23627851 - 2365440*Pi**2 + 18*jy**2*(-8108759 + 1030400*Pi**2) + 3*jz**2*(12451517 + 3964800*Pi**2)) - 2*ey**2*(180799920*jx*jy*Pi + jy**2*(2566775251 - 97708800*Pi**2) + 12*jx**2*(13671867 + 2965760*Pi**2) - 63*(3377491 + 15360*Pi**2 + jz**2*(-18347693 + 63360*Pi**2)))) + ez*(29169000*jx*jy**4*Pi + 5040*jx*jy**2*(-40762 - 610451*ey**2 + 50875*jx**2 - 356290*jz**2)*Pi + 2520*jx*(259455*ey**4 + 45295*jx**4 + 2*ey**2*(88916 + 127561*jx**2 - 18730*jz**2) + 8*(845 + 24782*jz**2) - 4*jx**2*(11009 + 131655*jz**2))*Pi + 5*jy**5*(-59206279 + 26476800*Pi**2) - 2*jy**3*(-112175778 + 60211200*Pi**2 + 1000*jz**2*(-195065 + 27216*Pi**2) - 245*jx**2*(-327391 + 441600*Pi**2) + ey**2*(-756949169 + 135340800*Pi**2)) + jy*(175*jx**4*(622579 + 326400*Pi**2) - 9*ey**4*(-493005977 + 34666240*Pi**2) + 8*(-11297 + 1451520*Pi**2 + 130*jz**2*(122617 + 47040*Pi**2)) + 4*jx**2*(-99*(51337 + 179200*Pi**2) + 10*jz**2*(-20962747 + 7308000*Pi**2)) + ey**2*(jx**2*(3735243994 - 752156160*Pi**2) + 4*(-317696002 + 59458560*Pi**2 + jz**2*(1179615359 + 110201280*Pi**2)))))))*eps_SA**3)/(2.9360128e7*Sqrt(ex**2 + ey**2 + ez**2))
            
        Delta_i = 0.0

    else: ### hyperbolic orbits
        raise RuntimeError("Third order hyperbolic orbits not implemented.")

    return Delta_e_TO,Delta_i

def compute_TO_prediction_averaged_over_argument_of_periapsis(eps_SA,eps_oct,e_per,e,i,Omega):
    Pi = np.pi
    Cos = np.cos
    Sin = np.sin
    Sqrt = np.sqrt
    Delta_e_mean = (9*e*Pi**2*eps_SA**2*(124 - 299*e**2 + 4*(-56 + 81*e**2)*Cos(2*i) + (36 + 39*e**2)*Cos(4*i)))/512.
    Delta_e_rms = (15*e*Pi*eps_SA*(32*Sqrt(1 - e**2)*Sin(i)**2 - 5*(-1 + e**2)*eps_SA*((5*Cos(i) + 3*Cos(3*i))*Cos(2*Omega) + 6*Sin(i)*Sin(2*i))))/(128.*Sqrt(2))

    Delta_e_mean += eps_SA**3*((-9*e*Sqrt(1 - e**2)*Pi*Cos(i)*(640*Pi + 3360*e**2*Pi + 40*(-4 + 49*e**2)*Pi*Cos(2*i) - 120*(4 + 11*e**2)*Pi*Cos(4*i) + 240*Pi*Cos(4*i - 2*Omega) + 660*e**2*Pi*Cos(4*i - 2*Omega) - 320*Pi*Cos(2*(i - Omega)) - 880*e**2*Pi*Cos(2*(i - Omega)) + 160*Pi*Cos(2*Omega) + 440*e**2*Pi*Cos(2*Omega) - 320*Pi*Cos(2*(i + Omega)) - 880*e**2*Pi*Cos(2*(i + Omega)) + 240*Pi*Cos(2*(2*i + Omega)) + 660*e**2*Pi*Cos(2*(2*i + Omega)) + 64*Sin(2*(i - 2*Omega)) + 176*e**2*Sin(2*(i - 2*Omega)) - 364*Sin(4*i - 2*Omega) - 1001*e**2*Sin(4*i - 2*Omega) + 616*Sin(2*(i - Omega)) + 854*e**2*Sin(2*(i - Omega)) - 16*Sin(4*(i - Omega)) - 44*e**2*Sin(4*(i - Omega)) + 504*Sin(2*Omega) - 294*e**2*Sin(2*Omega) + 96*Sin(4*Omega) + 264*e**2*Sin(4*Omega) - 616*Sin(2*(i + Omega)) - 854*e**2*Sin(2*(i + Omega)) + 16*Sin(4*(i + Omega)) + 44*e**2*Sin(4*(i + Omega)) + 364*Sin(2*(2*i + Omega)) + 1001*e**2*Sin(2*(2*i + Omega)) - 64*Sin(2*(i + 2*Omega)) - 176*e**2*Sin(2*(i + 2*Omega))))/4096.)
    Delta_e_rms += eps_SA**3*(3*e*Sqrt(2 - 2*e**2)*Pi*(1.0/np.sin(i)**2)*(-451600 + 210200*e**2 + 241400*e**4 - 1392144*Pi**2 + 7626288*e**2*Pi**2 - 8084769*e**4*Pi**2 + 247800*Cos(2*i) + 585900*e**2*Cos(2*i) - 833700*e**4*Cos(2*i) + 2228736*Pi**2*Cos(2*i) - 11383872*e**2*Pi**2*Cos(2*i) + 10460136*e**4*Pi**2*Cos(2*i) - 479600*Cos(4*i) + 476200*e**2*Cos(4*i) + 3400*e**4*Cos(4*i) - 1300800*Pi**2*Cos(4*i) + 4372800*e**2*Pi**2*Cos(4*i) - 2104500*e**4*Pi**2*Cos(4*i) + 43400*Cos(6*i) + 7700*e**2*Cos(6*i) - 51100*e**4*Cos(6*i) + 351744*Pi**2*Cos(6*i) - 228288*e**2*Pi**2*Cos(6*i) - 348456*e**4*Pi**2*Cos(6*i) - 34992*Pi**2*Cos(8*i) - 92016*e**2*Pi**2*Cos(8*i) - 69867*e**4*Pi**2*Cos(8*i) + 10000*Cos(2*i - 4*Omega) + 53800*e**2*Cos(2*i - 4*Omega) - 63800*e**4*Cos(2*i - 4*Omega) + 142560*Cos(4*i - 4*Omega) - 375120*e**2*Cos(4*i - 4*Omega) + 232560*e**4*Cos(4*i - 4*Omega) + 6640*Cos(6*i - 4*Omega) + 18520*e**2*Cos(6*i - 4*Omega) - 25160*e**4*Cos(6*i - 4*Omega) + 104440*Cos(2*i - 2*Omega) + 503020*e**2*Cos(2*i - 2*Omega) - 607460*e**4*Cos(2*i - 2*Omega) - 93520*Cos(4*i - 2*Omega) + 73640*e**2*Cos(4*i - 2*Omega) + 19880*e**4*Cos(4*i - 2*Omega) + 29960*Cos(6*i - 2*Omega) + 202580*e**2*Cos(6*i - 2*Omega) - 232540*e**4*Cos(6*i - 2*Omega) - 81760*Cos(2*Omega) - 1558480*e**2*Cos(2*Omega) + 1640240*e**4*Cos(2*Omega) + 321600*Cos(4*Omega) - 674400*e**2*Cos(4*Omega) + 352800*e**4*Cos(4*Omega) + 104440*Cos(2*(i + Omega)) + 503020*e**2*Cos(2*(i + Omega)) - 607460*e**4*Cos(2*(i + Omega)) + 142560*Cos(4*(i + Omega)) - 375120*e**2*Cos(4*(i + Omega)) + 232560*e**4*Cos(4*(i + Omega)) - 93520*Cos(4*i + 2*Omega) + 73640*e**2*Cos(4*i + 2*Omega) + 19880*e**4*Cos(4*i + 2*Omega) + 29960*Cos(6*i + 2*Omega) + 202580*e**2*Cos(6*i + 2*Omega) - 232540*e**4*Cos(6*i + 2*Omega) + 10000*Cos(2*i + 4*Omega) + 53800*e**2*Cos(2*i + 4*Omega) - 63800*e**4*Cos(2*i + 4*Omega) + 6640*Cos(6*i + 4*Omega) + 18520*e**2*Cos(6*i + 4*Omega) - 25160*e**4*Cos(6*i + 4*Omega) + 55200*Pi*Sin(2*i - 2*Omega) + 27600*e**2*Pi*Sin(2*i - 2*Omega) - 82800*e**4*Pi*Sin(2*i - 2*Omega) - 62400*Pi*Sin(4*i - 2*Omega) - 247200*e**2*Pi*Sin(4*i - 2*Omega) + 309600*e**4*Pi*Sin(4*i - 2*Omega) - 55200*Pi*Sin(6*i - 2*Omega) + 68400*e**2*Pi*Sin(6*i - 2*Omega) - 13200*e**4*Pi*Sin(6*i - 2*Omega) - 124800*Pi*Sin(2*Omega) - 302400*e**2*Pi*Sin(2*Omega) + 427200*e**4*Pi*Sin(2*Omega) - 55200*Pi*Sin(2*(i + Omega)) - 27600*e**2*Pi*Sin(2*(i + Omega)) + 82800*e**4*Pi*Sin(2*(i + Omega)) + 62400*Pi*Sin(4*i + 2*Omega) + 247200*e**2*Pi*Sin(4*i + 2*Omega) - 309600*e**4*Pi*Sin(4*i + 2*Omega) + 55200*Pi*Sin(6*i + 2*Omega) - 68400*e**2*Pi*Sin(6*i + 2*Omega) + 13200*e**4*Pi*Sin(6*i + 2*Omega)))/(2.62144e6*(-1 + e**2))

    return Delta_e_mean,Delta_e_rms
