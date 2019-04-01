"""
Script to numerically integrate the equations (EOM) for a binary perturbed by a passing body on a hyperbolic orbit.
The EOM are averaged over the binary orbit, but not over the perturber's orbit. 
Valid to the lowest expansion order in r_bin/r_per (i.e., second order or quadrupole order).

Adrian Hamers
March 2019
"""

import argparse
import numpy as np
import os

import pickle

from scipy.integrate import odeint

try:
    from matplotlib import pyplot
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def add_bool_arg(parser, name, default=False,help=None):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true',help="Enable %s"%help)
    group.add_argument('--no-' + name, dest=name, action='store_false',help="Disable %s"%help)
    parser.set_defaults(**{name:default})

def parse_arguments():
    

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",                           type=int,       dest="mode",                        default=1,              help="mode -- 1: single integration; 2: single integration (illustrating Fourier series); 3: series integration; 4: series integration (with detailed time plots)")
    parser.add_argument("--name",                           type=str,       dest="name",                        default="test01",       help="name")
    parser.add_argument("--m1",                             type=float,     dest="m1",                          default=1.0,            help="Primary mass")
    parser.add_argument("--m2",                             type=float,     dest="m2",                          default=1.0,            help="Secondary mass")
    parser.add_argument("--M_per",                          type=float,     dest="M_per",                       default=1.0,            help="Perturber mass")
    parser.add_argument("--e_per",                          type=float,     dest="e_per",                       default=1.5+1.0e-15,    help="Perturber eccentricity")
    parser.add_argument("--Q",                              type=float,     dest="Q",                           default=4.0,            help="Perturber periapsis distance (same units as a)")
    parser.add_argument("--Q_min",                          type=float,     dest="Q_min",                       default=2.0,            help="Minimum perturber periapsis distance (in case of series)")
    parser.add_argument("--Q_max",                          type=float,     dest="Q_max",                       default=50.0,           help="Maximum perturber periapsis distance (in case of series)")
    parser.add_argument("--N_Q",                            type=int,       dest="N_Q",                         default=200,            help="Number of systems in series")
    parser.add_argument("--a",                              type=float,     dest="a",                           default=1.0,            help="Binary semimajor axis (same units as Q)")    
    parser.add_argument("--e",                              type=float,     dest="e",                           default=0.999,          help="Binary eccentricity")    
    parser.add_argument("--i",                              type=float,     dest="i",                           default=np.pi/2.0,      help="Binary inclination")    
    parser.add_argument("--omega",                          type=float,     dest="omega",                       default=np.pi/4.0,      help="Binary argument of periapsis")
    parser.add_argument("--Omega",                          type=float,     dest="Omega",                       default=-np.pi/4.0,     help="Binary longitude of the ascending node")
    parser.add_argument("--N_steps",                        type=int,       dest="N_steps",                     default=100,            help="Number of external output steps taken by odeint")
    parser.add_argument("--mxstep",                         type=int,       dest="mxstep",                      default=1000000,        help="Maximum number of internal steps taken in the ODE integration. Increase if ODE integrator give mstep errors. ")    
    parser.add_argument("--theta_bin",                      type=float,     dest="theta_bin",                   default=1.0,            help="Initial binary true anomaly (3-body integration only)")
    parser.add_argument("--fraction_theta_0",               type=float,     dest="fraction_theta_0",            default=0.9,            help="Initial perturber true anomaly (3-body only), expressed as a fraction of -\arccos(-1/e_per). Default=0.9; increase if 3-body integrations do not seem converged. ")
    parser.add_argument("--G",                              type=float,     dest="G",                           default=1.0,            help="Gravitational constant used in 3-body integrations. Should not affect results. ")
    parser.add_argument("--fontsize",                       type=float,     dest="fontsize",                    default=22,             help="Fontsize for plots")
    parser.add_argument("--labelsize",                      type=float,     dest="labelsize",                   default=16,             help="Labelsize for plots")

    ### boolean arguments ###
    add_bool_arg(parser, 'verbose',                         default=False,          help="Verbose terminal output")
    add_bool_arg(parser, 'calc',                            default=True,           help="Do calculation (and save results). If False, will try to load previous results")
    add_bool_arg(parser, 'plot',                            default=True,           help="Make plots")
    add_bool_arg(parser, 'plot_fancy',                      default=False,          help="Use LaTeX for plot labels (slower)")
    add_bool_arg(parser, 'show',                            default=True,           help="Show plots")
    add_bool_arg(parser, 'include_quadrupole_terms',        default=True,           help="Include quadrupole-order terms")
    add_bool_arg(parser, 'include_octupole_terms',          default=False,          help="include octupole-order terms")
    add_bool_arg(parser, 'do_nbody',                        default=False,          help="Do 3-body integrations as well as SA")
    
    args = parser.parse_args()
               
    args.m = args.m1 + args.m2
    args.data_filename = os.path.dirname(os.path.realpath(__file__)) + '/data/data_' + str(args.name) + '_m1_' + str(args.m1) + '_m2_' + str(args.m2) + '_M_per_' + str(args.M_per) + '_e_per_' \
        + str(args.e_per) + '_Q_' + str(args.Q) + '_a_' + str(args.a) + '_e_' + str(args.e) + '_i_' + str(args.i) + '_omega_' + str(args.omega) + '_Omega_' + str(args.Omega) \
        + '_do_nbody_' + str(args.do_nbody) + '.pkl'
    args.series_data_filename = os.path.dirname(os.path.realpath(__file__)) + '/data/series_data_' + str(args.name) + '_m1_' + str(args.m1) + '_m2_' + str(args.m2) + '_M_per_' \
        + str(args.M_per) + '_e_per_' + str(args.e_per) + '_Q_' + str(args.Q) + '_a_' + str(args.a) + '_e_' + str(args.e) + '_i_' + str(args.i) + '_omega_' + str(args.omega) + '_Omega_' \
        + str(args.Omega) + '_Q_min_' + str(args.Q_min) + '_Q_max_' + str(args.Q_max) + '_N_Q_' + str(args.N_Q) \
        + '_do_nbody_' + str(args.do_nbody) + '.pkl'

    return args
    

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
    
#    v_sq = numpy.sum([x**2 for x in r_dot_per_vec])
#    r = numpy.sqrt( numpy.sum([x**2 for x in r_per_vec]) )
#    v_infty = numpy.sqrt( v_sq - 2.0*constants.G*M_tot/r)
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

    i = args.i
    omega = args.omega
    Omega = args.Omega
    
    ex,ey,ez,jx,jy,jz = orbital_elements_to_orbital_vectors(e,i,omega,Omega)
        
    N_steps = args.N_steps
    theta_0 = np.arccos(-1.0/e_per)
    thetas = np.linspace(-theta_0, theta_0, N_steps)

    ODE_args = (eps_SA,eps_oct,e_per,args)

    RHR_vec = [ex,ey,ez,jx,jy,jz]
        
    if args.verbose==True:
        print 'eps_SA',eps_SA
        print 'RHR_vec',RHR_vec
    
    ### numerical solution ###
    sol = odeint(RHS_function, RHR_vec, thetas, args=ODE_args,mxstep=args.mxstep)
    
    ex_sol = np.array(sol[:,0])
    ey_sol = np.array(sol[:,1])
    ez_sol = np.array(sol[:,2])
    jx_sol = np.array(sol[:,3])
    jy_sol = np.array(sol[:,4])
    jz_sol = np.array(sol[:,5])

    e_sol = [np.sqrt(ex_sol[i]**2 + ey_sol[i]**2 + ez_sol[i]**2) for i in range(len(thetas))]
    j_sol = [np.sqrt(jx_sol[i]**2 + jy_sol[i]**2 + jz_sol[i]**2) for i in range(len(thetas))]
    i_sol = [np.arccos(jz_sol[i]/j_sol[i]) for i in range(len(thetas))]
    
    Delta_e = e_sol[-1] - e_sol[0]
    Delta_i = i_sol[-1] - i_sol[0]

    nbody_a_sol,nbody_e_sol,nbody_i_sol,nbody_Delta_a,nbody_Delta_e,nbody_Delta_i,nbody_energy_errors_sol = None,None,None,None,None,None,None
    if args.do_nbody==True:
        nbody_a_sol,nbody_e_sol,nbody_i_sol,nbody_Delta_a,nbody_Delta_e,nbody_Delta_i,nbody_energy_errors_sol = integrate_nbody(args)
        print 'Q/a',args.Q/args.a
        print 'SA Delta e',Delta_e,'Delta i',Delta_i
        print 'nbody Delta e',nbody_Delta_e,'Delta i',nbody_Delta_i,'nbody energy error',nbody_energy_errors_sol[-1]

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
    e_per = args.e_per
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
    #a = args.fraction_theta_0*np.arccos(-1.0/e_per)
    
    a = -theta_0
    tend = (1.0/n_per)*( -4*np.arctanh(((-1 + e_per)*np.tan(a/2.))/np.sqrt(-1 + e_per**2)) + (2*e_per*np.sqrt(-1 + e_per**2)*np.sin(a))/(1 + e_per*np.cos(a)) )
    
    #print 'tend',tend,'P_bin',2.0*np.pi*np.sqrt(a**3/(G*M)),(1.0/n_per)
    
    times = np.linspace(0.0,tend,N_steps)

    ODE_args = (G,m1,m2,M_per,args)

    RHR_vec = [R1[0],R1[1],R1[2],V1[0],V1[1],V1[2],R2[0],R2[1],R2[2],V2[0],V2[1],V2[2],R3[0],R3[1],R3[2],V3[0],V3[1],V3[2]]

    if args.verbose==True:
        print 'eps_SA',eps_SA
        print 'RHR_vec',RHR_vec
    
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


def integrate_series(args):
    Q_points = pow(10.0,np.linspace(np.log10(args.Q_min),np.log10(args.Q_max),args.N_Q))
    Delta_es = []
    Delta_is = []
    
    nbody_Delta_es = []
    nbody_Delta_is = []
    do_nbody = False
    thetas_Q = []
    es_Q = []
    is_Q = []
    for index_Q,Q in enumerate(Q_points):
        args.Q = Q
        data = integrate(args)
        Delta_e = data["Delta_e"]
        Delta_i = data["Delta_i"]
        
        thetas_Q.append( data["thetas"] )
        es_Q.append( data["e_sol"] )
        is_Q.append( data["i_sol"] )
        if data["do_nbody"] == True:
            nbody_Delta_es.append(data["nbody_Delta_e"])
            nbody_Delta_is.append(data["nbody_Delta_i"])
            do_nbody = True
        
        Delta_es.append(Delta_e)
        Delta_is.append(Delta_i)

    thetas_Q = np.array(thetas_Q)
    es_Q = np.array(es_Q)
    is_Q = np.array(is_Q)
    
    Delta_es = np.array(Delta_es)
    Delta_is = np.array(Delta_is)
    nbody_Delta_es = np.array(nbody_Delta_es)
    nbody_Delta_is = np.array(nbody_Delta_is)
    
    data_series = {'Q_points':Q_points,'Delta_es':Delta_es,'Delta_is':Delta_is,'do_nbody':do_nbody,'nbody_Delta_es':nbody_Delta_es,'nbody_Delta_is':nbody_Delta_is,'thetas_Q':thetas_Q,'es_Q':es_Q,'is_Q':is_Q}

    filename = args.series_data_filename

    with open(filename, 'wb') as handle:
        pickle.dump(data_series, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data_series
    
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
    
def compute_eps_oct(m1,m2,m,a_div_Q,e_per):
    return a_div_Q*np.fabs(m1-m2)/((1.0+e_per)*m)
    
def compute_DA_prediction(args,eps_SA,e_per,ex,ey,ez,jx,jy,jz):
    
    Delta_e = (5*eps_SA*(np.sqrt(1 - e_per**(-2))*((1 + 2*e_per**2)*ey*ez*jx + (1 - 4*e_per**2)*ex*ez*jy + 2*(-1 + e_per**2)*ex*ey*jz) + 3*e_per*ez*(ey*jx - ex*jy)*np.arccos(-1.0/e_per)))/(2.*e_per*np.sqrt(ex**2 + ey**2 + ez**2))

    ArcCos = np.arccos
    Sqrt = np.sqrt
    eper = e_per
    Pi = np.pi

    Delta_i = -ArcCos(jz/Sqrt(jx**2 + jy**2 + jz**2)) + ArcCos((jz - (Sqrt(1 - eper**(-2))*(-1 + eper**2)*(5*ex*ey - jx*jy)*eps_SA)/eper)/Sqrt((jz - (Sqrt(1 - eper**(-2))*(-1 + eper**2)*(5*ex*ey - jx*jy)*eps_SA)/eper)**2 + (jx - ((5*ey*ez - jy*jz)*eps_SA*(Sqrt(1 - eper**(-2))*(1 + 2*eper**2) + 3*eper*ArcCos(-(1/eper))))/(2.*eper))**2 + (jy + ((5*ex*ez - jx*jz)*eps_SA*(Sqrt(1 - eper**(-2))*(-1 + 4*eper**2) + 3*eper*ArcCos(-(1/eper))))/(2.*eper))**2))
  
    return Delta_e,Delta_i



def compute_DA_prediction_oct(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz):

    ArcCos = np.arccos
    Sqrt = np.sqrt
    eper = e_per
    Pi = np.pi
    Delta_e = -(5*eps_oct*eps_SA*(Sqrt(1 - eper**(-2))*(ez*jy*(14*ey**2 + 6*jx**2 - 2*jy**2 + 8*eper**4*(-1 + ey**2 + 8*ez**2 + 2*jx**2 + jy**2) + eper**2*(-4 - 31*ey**2 + 32*ez**2 - 7*jx**2 + 9*jy**2)) - ey*(2*(7*ey**2 + jx**2 - jy**2) + 8*eper**4*(-1 + ey**2 + 8*ez**2 + 4*jx**2 + jy**2) + eper**2*(-4 - 31*ey**2 + 32*ez**2 + 11*jx**2 + 9*jy**2))*jz + ex**2*(-((14 + 45*eper**2 + 160*eper**4)*ez*jy) + 3*(14 - 27*eper**2 + 16*eper**4)*ey*jz) + 2*(-2 + 9*eper**2 + 8*eper**4)*ex*jx*(7*ey*ez + jy*jz)) + 3*eper**3*(ez*jy*(-4 - 3*ey**2 + 32*ez**2 + 5*jx**2 + 5*jy**2) + ey*(4 + 3*ey**2 - 32*ez**2 - 15*jx**2 - 5*jy**2)*jz + ex**2*(-73*ez*jy + 3*ey*jz) + 10*ex*jx*(7*ey*ez + jy*jz))*ArcCos(-1.0/eper)))/(32.*eper**2*Sqrt(ex**2 + ey**2 + ez**2))
    return Delta_e,0.0

def f(n,e_per):
    L = np.arccos(-1.0/e_per)
    npi = n*np.pi
    return (npi - 3.0*L)*(npi - 2.0*L)*(npi - L)*(npi + L)*(npi + 2.0*L)*(npi + 3.0*L)

def compute_CDA_prediction(args,eps_SA,e_per,ex,ey,ez,jx,jy,jz):


    Delta_e_DA,Delta_i_CDA = compute_DA_prediction(args,eps_SA,e_per,ex,ey,ez,jx,jy,jz)

    f1 = f(1.0,e_per)
    f2 = f(2.0,e_per)
    f3 = f(3.0,e_per)

    #print 'fs',f1,f2,f3
    Delta_i = 0.0

    ArcCos = np.arccos
    Sqrt = np.sqrt
    eper = e_per
    Pi = np.pi
    
    epsilon = 1.0e-5
    if e_per <= 1.0 + epsilon: ### parabolic limit
        Delta_e_CDA = eps_SA*(15*ez*(ey*jx - ex*jy)*np.pi)/(2.*np.sqrt(ex**2 + ey**2 + ez**2)) \
            + eps_SA**2*(-9*(-6*ez**2*(jx**2 + jy**2) - 4*ey*ez*jy*jz + 4*ex*jx*(3*ey*jy - ez*jz) + ey**2*(25*ez**2 - 6*jx**2 + jz**2) + ex**2*(25*ez**2 - 6*jy**2 + jz**2))*np.pi**2)/(8.*np.sqrt(ex**2 + ey**2 + ez**2))

    else:
        exn = ex + (eps_SA*(np.sqrt(1 - e_per**(-2))*(ez*(jy - 10*e_per**2*jy) + (-5 + 2*e_per**2)*ey*jz) - 3*e_per*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per)))/(4.*e_per) - (3*np.sqrt(1 - e_per**(-2))*(ez*jx - 5*ex*jz)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)
        eyn = ey + (eps_SA*(np.sqrt(1 - e_per**(-2))*(ez*(jx + 8*e_per**2*jx) + (-5 + 8*e_per**2)*ex*jz) + 3*e_per*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per)))/(4.*e_per) + (3*np.sqrt(1 - e_per**(-2))*(ez*jy - 5*ey*jz)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)
        ezn = ez + (np.sqrt(1 - e_per**(-2))*((2 + e_per**2)*ey*jx + (2 - 5*e_per**2)*ex*jy)*eps_SA + 3*e_per*(ey*jx - ex*jy)*eps_SA*np.arccos(-(1/e_per)))/(2.*e_per) - (12*np.sqrt(1 - e_per**(-2))*(ex*jx - ey*jy)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)
        jxn = jx - ((5*ey*ez - jy*jz)*eps_SA*(np.sqrt(1 - e_per**(-2))*(1 + 2*e_per**2) + 3*e_per*np.arccos(-(1/e_per))))/(4.*e_per) + (3*np.sqrt(1 - e_per**(-2))*(5*ex*ez - jx*jz)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)
        jyn = jy + ((5*ex*ez - jx*jz)*eps_SA*(np.sqrt(1 - e_per**(-2))*(-1 + 4*e_per**2) + 3*e_per*np.arccos(-(1/e_per))))/(4.*e_per) - (3*np.sqrt(1 - e_per**(-2))*(5*ey*ez - jy*jz)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)
        jzn = jz - (np.sqrt(1 - e_per**(-2))*(-1 + e_per**2)*(5*ex*ey - jx*jy)*eps_SA)/(2.*e_per) - (3*np.sqrt(1 - e_per**(-2))*(5*ex**2 - 5*ey**2 - jx**2 + jy**2)*eps_SA*np.arccos(-(1/e_per))**3*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*e_per**2)*np.pi**2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3)

        Delta_e_CDA = compute_DA_prediction(args,eps_SA,e_per,exn,eyn,ezn,jxn,jyn,jzn)[0]

        lmax=3
        if lmax==1:
            Delta_e_CDA += (3*eps_SA**2*np.arccos(-(1/e_per))*((5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(60*(-1 + e_per**2)*((-1 + e_per**2)*ex**2*jx*jy + ey**2*jx*jy - e_per**2*ez**2*jx*jy - 2*(-2 + e_per**2)*ey*ez*jx*jz + 2*(-2 + e_per**2)*ex*ez*jy*jz + ex*ey*(-((-1 + e_per**2)*jx**2) - jy**2 + e_per**2*jz**2))*np.pi**4*np.arccos(-(1/e_per))**3 - 60*(-1 + e_per**2)*((5 + 7*e_per**2)*ex**2*jx*jy + (-5 + 6*e_per**2)*ey**2*jx*jy - 13*e_per**2*ez**2*jx*jy - 2*(10 + e_per**2)*ey*ez*jx*jz + 2*(10 + e_per**2)*ex*ez*jy*jz + ex*ey*(-((5 + 7*e_per**2)*jx**2) + (5 - 6*e_per**2)*jy**2 + 13*e_per**2*jz**2))*np.pi**2*np.arccos(-(1/e_per))**5 + 360*(-1 + e_per**2)*((1 + 2*e_per**2)*ex**2*jx*jy + (-1 + 4*e_per**2)*ey**2*jx*jy - 6*e_per**2*ez**2*jx*jy + 4*(-1 + e_per**2)*ey*ez*jx*jz - 4*(-1 + e_per**2)*ex*ez*jy*jz + ex*ey*(-((1 + 2*e_per**2)*jx**2) + (1 - 4*e_per**2)*jy**2 + 6*e_per**2*jz**2))*np.arccos(-(1/e_per))**7 - 3*np.sqrt(1 - e_per**(-2))*e_per**3*(2*ex**2*jx*jy + ey*jx*(2*ey*jy - 5*ez*jz) + ex*(50*ey*ez**2 - 2*ey*(jx**2 + jy**2) - 5*ez*jy*jz))*np.arccos(-(1/e_per))*np.arccos(-1.0/e_per)*f1 + np.arccos(-1.0/e_per)*(2*(-1 + e_per**2)*(25*(-1 + e_per**2)*ex**3*ey + (7 - 10*e_per**2)*ex**2*jx*jy + jx*((-7 + 4*e_per**2)*ey**2*jy - 36*e_per**2*ez**2*jy + 2*(-5 + 17*e_per**2)*ey*ez*jz) + ex*(-25*(-1 + e_per**2)*ey**3 + 2*ey*(jx**2 - jy**2) + 2*(5 + 7*e_per**2)*ez*jy*jz + e_per**2*ey*(-75*ez**2 + jx**2 + 5*jy**2 + 15*jz**2))) - 3*np.sqrt(1 - e_per**(-2))*e_per**3*(24*ez**2*jx*jy - 11*ez*(ey*jx + ex*jy)*jz - 10*ex*ey*jz**2)*np.arccos(-1.0/e_per))*f1))/(2.*e_per**4*np.sqrt(ex**2 + ey**2 + ez**2)*f1**2)
        if lmax==2:
            Delta_e_CDA += (3*eps_SA**2*np.arccos(-(1/e_per))*np.arccos(-1.0/e_per)*((ey*(-12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*(-8*(2*jx*((7 - 4*e_per**2)*ey*jy + (-2 + 5*e_per**2)*ez*jz) + ex*(25*(-2 + e_per**2)*ey**2 - 5*ez**2 + 4*jy**2 - e_per**2*jy**2 + 25*jz**2 - 10*e_per**2*jz**2))*np.pi**4 + 2*(2*jx*(-5*(7 + 2*e_per**2)*ey*jy + (10 + 53*e_per**2)*ez*jz) + ex*(25*(10 + e_per**2)*ey**2 - 5*(-5 + 6*e_per**2)*ez**2 - 20*jy**2 + 11*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 3*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.arccos(-(1/e_per))**4)*f1**2 - 12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(-8*(5*(-5 + 2*e_per**2)*ex**3 + 4*jx*((-1 + 2*e_per**2)*ey*jy - ez*jz) + ex*(-5*(-5 + 2*e_per**2)*ey**2 + 5*(-1 + 2*e_per**2)*ez**2 + 9*jx**2 - 10*e_per**2*jx**2 - 5*jy**2 + 2*e_per**2*jy**2 + 25*jz**2 - 10*e_per**2*jz**2))*np.pi**5 + 2*(-5*(-25 + 4*e_per**2)*ex**3 + 4*jx*(5*ey*(jy + 4*e_per**2*jy) + (5 - 6*e_per**2)*ez*jz) + ex*(5*(-25 + 4*e_per**2)*ey**2 + 25*(1 + 4*e_per**2)*ez**2 - 45*jx**2 - 76*e_per**2*jx**2 + 25*jy**2 - 4*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 3*(5*(-5 + 8*e_per**2)*ex**3 - 4*jx*(ey*(jy + 8*e_per**2*jy) + (1 - 4*e_per**2)*ez*jz) + ex*(-5*(-5 + 8*e_per**2)*ey**2 - 5*(1 + 8*e_per**2)*ez**2 + 9*jx**2 + 24*e_per**2*jx**2 - 5*jy**2 + 8*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f1**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*(4*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2)) + 3*e_per*(5*ex*ez**2 + 2*ey*jx*jy - 2*ex*jy**2 - ez*jx*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per))*f1**2*f2 - 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*((2*jx*((-7 + 4*e_per**2)*ey*jy + (2 - 5*e_per**2)*ez*jz) + ex*(-25*(-2 + e_per**2)*ey**2 + 5*ez**2 - 4*jy**2 + e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi**4 + (2*jx*(-5*(7 + 2*e_per**2)*ey*jy + (10 + 53*e_per**2)*ez*jz) + ex*(25*(10 + e_per**2)*ey**2 - 5*(-5 + 6*e_per**2)*ez**2 - 20*jy**2 + 11*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 6*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.arccos(-(1/e_per))**4)*f2**2 - 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*((-5*(-5 + 2*e_per**2)*ex**3 + 4*jx*(ey*(jy - 2*e_per**2*jy) + ez*jz) + ex*(5*(-5 + 2*e_per**2)*ey**2 + (5 - 10*e_per**2)*ez**2 - 9*jx**2 + 10*e_per**2*jx**2 + 5*jy**2 - 2*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi**5 + (-5*(-25 + 4*e_per**2)*ex**3 + 4*jx*(5*ey*(jy + 4*e_per**2*jy) + (5 - 6*e_per**2)*ez*jz) + ex*(5*(-25 + 4*e_per**2)*ey**2 + 25*(1 + 4*e_per**2)*ez**2 - 45*jx**2 - 76*e_per**2*jx**2 + 25*jy**2 - 4*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 6*(5*(-5 + 8*e_per**2)*ex**3 - 4*jx*(ey*(jy + 8*e_per**2*jy) + (1 - 4*e_per**2)*ez*jz) + ex*(-5*(-5 + 8*e_per**2)*ey**2 - 5*(1 + 8*e_per**2)*ez**2 + 9*jx**2 + 24*e_per**2*jx**2 - 5*jy**2 + 8*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f2**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*((-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2)) + 3*e_per*(5*ex*ez**2 + 2*ey*jx*jy - 2*ex*jy**2 - ez*jx*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per))*f1*f2**2))/np.pi + (ex*(12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*(-8*(25*(-2 + e_per**2)*ex**2*ey + 2*(7 - 3*e_per**2)*ex*jx*jy - 2*(2 + 3*e_per**2)*ez*jy*jz + ey*(5*(-1 + e_per**2)*ez**2 + (4 - 3*e_per**2)*jx**2 + 5*(5 - 3*e_per**2)*jz**2))*np.pi**4 + 2*(25*(10 + e_per**2)*ex**2*ey + 2*(-35 + 3*e_per**2)*ex*jx*jy + 2*(10 - 51*e_per**2)*ez*jy*jz + 5*ey*((5 + 7*e_per**2)*ez**2 - (4 + 3*e_per**2)*jx**2 - (25 + 9*e_per**2)*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 3*(50*(-1 + e_per**2)*ex**2*ey + 2*(7 - 10*e_per**2)*ex*jx*jy + 4*(-1 + 13*e_per**2)*ez*jy*jz + ey*(-5*(1 + 2*e_per**2)*ez**2 + 2*(2 + e_per**2)*jx**2 + 5*(5 - 2*e_per**2)*jz**2))*np.arccos(-(1/e_per))**4)*f1**2 - 12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(-8*(5*(-5 + 3*e_per**2)*ex**2*ey - 5*(-5 + 3*e_per**2)*ey**3 + 4*(1 + e_per**2)*ex*jx*jy - 4*(-1 + e_per**2)*ez*jy*jz + ey*(5*(1 + e_per**2)*ez**2 + (5 - 3*e_per**2)*jx**2 - 9*jy**2 - e_per**2*jy**2 - 25*jz**2 + 15*e_per**2*jz**2))*np.pi**5 + 2*(5*(25 + 9*e_per**2)*ex**2*ey - 5*(25 + 9*e_per**2)*ey**3 + 4*(-5 + 19*e_per**2)*ex*jx*jy - 4*(5 + 7*e_per**2)*ez*jy*jz + ey*(5*(-5 + 19*e_per**2)*ez**2 - (25 + 9*e_per**2)*jx**2 + 45*jy**2 - 67*e_per**2*jy**2 + 125*jz**2 + 45*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 3*(5*(-5 + 2*e_per**2)*ex**2*ey - 5*(-5 + 2*e_per**2)*ey**3 + 4*(1 - 10*e_per**2)*ex*jx*jy + 4*(1 + 2*e_per**2)*ez*jy*jz + ey*((5 - 50*e_per**2)*ez**2 + (5 - 2*e_per**2)*jx**2 - 9*jy**2 + 42*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f1**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*(4*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-50*(-1 + e_per**2)*ex**2*ey + 2*(-7 + 10*e_per**2)*ex*jx*jy + 4*(1 - 13*e_per**2)*ez*jy*jz + ey*(5*(1 + 2*e_per**2)*ez**2 - 2*(2 + e_per**2)*jx**2 + 5*(-5 + 2*e_per**2)*jz**2)) + 3*e_per*(5*ey*ez**2 - 2*ey*jx**2 + 2*ex*jx*jy - ez*jy*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per))*f1**2*f2 + 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*((-25*(-2 + e_per**2)*ex**2*ey + 2*(-7 + 3*e_per**2)*ex*jx*jy + 2*(2 + 3*e_per**2)*ez*jy*jz + ey*(-5*(-1 + e_per**2)*ez**2 + (-4 + 3*e_per**2)*jx**2 + 5*(-5 + 3*e_per**2)*jz**2))*np.pi**4 + (25*(10 + e_per**2)*ex**2*ey + 2*(-35 + 3*e_per**2)*ex*jx*jy + 2*(10 - 51*e_per**2)*ez*jy*jz + 5*ey*((5 + 7*e_per**2)*ez**2 - (4 + 3*e_per**2)*jx**2 - (25 + 9*e_per**2)*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 6*(50*(-1 + e_per**2)*ex**2*ey + 2*(7 - 10*e_per**2)*ex*jx*jy + 4*(-1 + 13*e_per**2)*ez*jy*jz + ey*(-5*(1 + 2*e_per**2)*ez**2 + 2*(2 + e_per**2)*jx**2 + 5*(5 - 2*e_per**2)*jz**2))*np.arccos(-(1/e_per))**4)*f2**2 - 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*((-5*(-5 + 3*e_per**2)*ex**2*ey + 5*(-5 + 3*e_per**2)*ey**3 - 4*(1 + e_per**2)*ex*jx*jy + 4*(-1 + e_per**2)*ez*jy*jz + ey*(-5*(1 + e_per**2)*ez**2 + (-5 + 3*e_per**2)*jx**2 + 9*jy**2 + e_per**2*jy**2 + 25*jz**2 - 15*e_per**2*jz**2))*np.pi**5 + (5*(25 + 9*e_per**2)*ex**2*ey - 5*(25 + 9*e_per**2)*ey**3 + 4*(-5 + 19*e_per**2)*ex*jx*jy - 4*(5 + 7*e_per**2)*ez*jy*jz + ey*(5*(-5 + 19*e_per**2)*ez**2 - (25 + 9*e_per**2)*jx**2 + 45*jy**2 - 67*e_per**2*jy**2 + 125*jz**2 + 45*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 6*(5*(-5 + 2*e_per**2)*ex**2*ey - 5*(-5 + 2*e_per**2)*ey**3 + 4*(1 - 10*e_per**2)*ex*jx*jy + 4*(1 + 2*e_per**2)*ez*jy*jz + ey*((5 - 50*e_per**2)*ez**2 + (5 - 2*e_per**2)*jx**2 - 9*jy**2 + 42*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f2**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*((-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-50*(-1 + e_per**2)*ex**2*ey + 2*(-7 + 10*e_per**2)*ex*jx*jy + 4*(1 - 13*e_per**2)*ez*jy*jz + ey*(5*(1 + 2*e_per**2)*ez**2 - 2*(2 + e_per**2)*jx**2 + 5*(-5 + 2*e_per**2)*jz**2)) + 3*e_per*(5*ey*ez**2 - 2*ey*jx**2 + 2*ex*jx*jy - ez*jy*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per))*f1*f2**2))/np.pi - 12*np.sqrt(1 - e_per**(-2))*e_per**2*ez*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*np.arccos(-1.0/e_per)*(2*np.sqrt(1 - e_per**(-2))*((ey*jx - ex*jy)*jz + e_per**2*(5*ex*ey*ez + 3*ez*jx*jy - ey*jx*jz + ex*jy*jz)) + e_per*(6*ez*jx*jy + ey*jx*jz + ex*jy*jz)*np.arccos(-1.0/e_per))*f1*f2*(f1 + f2) - e_per*(-5 + 2*e_per**2)*(10*ex*ey*ez - ey*jx*jz - ex*jy*jz)*np.pi**2*np.arccos(-(1/e_per))*np.arccos(-1.0/e_per)*f1*f2*(4*f1 + f2) - (-5 + 2*e_per**2)*np.pi**2*np.arccos(-1.0/e_per)*(2*np.sqrt(1 - e_per**(-2))*((ey*jx - ex*jy)*jz + e_per**2*(5*ex*ey*ez + 3*ez*jx*jy - ey*jx*jz + ex*jy*jz)) + e_per*(6*ez*jx*jy + ey*jx*jz + ex*jy*jz)*np.arccos(-1.0/e_per))*f1*f2*(4*f1 + f2) + 36*np.sqrt(1 - e_per**(-2))*(-5 + 8*e_per**2)*(4*(ey*jx - ex*jy)*jz + e_per**2*(5*ex*ey*ez + 5*ez*jx*jy - ey*jx*jz + 7*ex*jy*jz))*np.arccos(-(1/e_per))**9*(f1**2 + f2**2) - np.sqrt(1 - e_per**(-2))*(1320*(-(ey*jx*jz) + ex*jy*jz) + 16*e_per**4*(55*ex*ey*ez + 55*ez*jx*jy + 21*ey*jx*jz + 45*ex*jy*jz) - e_per**2*(1225*ex*ey*ez + 1225*ez*jx*jy - 1173*ey*jx*jz + 2643*ex*jy*jz))*np.pi**2*np.arccos(-(1/e_per))**7*(4*f1**2 + f2**2) + 2*np.sqrt(1 - e_per**(-2))*(e_per**4*(85*ex*ey*ez + 85*ez*jx*jy + 111*ey*jx*jz - 9*ex*jy*jz) + 240*(-(ey*jx*jz) + ex*jy*jz) - e_per**2*(175*ex*ey*ez + 175*ez*jx*jy + 141*ey*jx*jz + 69*ex*jy*jz))*np.pi**4*np.arccos(-(1/e_per))**5*(16*f1**2 + f2**2) + np.arccos(-(1/e_per))**3*(e_per*(-5 + 8*e_per**2)*(10*ex*ey*ez - ey*jx*jz - ex*jy*jz)*np.arccos(-1.0/e_per)*f1*f2*(f1 + f2) - np.sqrt(1 - e_per**(-2))*(-5 + 2*e_per**2)*(24*(-(ey*jx) + ex*jy)*jz + e_per**2*(5*ex*ey*ez + 5*ez*jx*jy + 15*ey*jx*jz - 9*ex*jy*jz))*np.pi**6*(64*f1**2 + f2**2)))))/(2.*e_per**4*np.sqrt(ex**2 + ey**2 + ez**2)*f1**2*f2**2)
        if lmax==3:
            Delta_e_CDA += (3*eps_SA**2*np.arccos(-(1/e_per))*np.arccos(-1.0/e_per)*((ey*(-18*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(9*(5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*(-27*(2*jx*((7 - 4*e_per**2)*ey*jy + (-2 + 5*e_per**2)*ez*jz) + ex*(25*(-2 + e_per**2)*ey**2 - 5*ez**2 + 4*jy**2 - e_per**2*jy**2 + 25*jz**2 - 10*e_per**2*jz**2))*np.pi**4 + 3*(2*jx*(-5*(7 + 2*e_per**2)*ey*jy + (10 + 53*e_per**2)*ez*jz) + ex*(25*(10 + e_per**2)*ey**2 - 5*(-5 + 6*e_per**2)*ez**2 - 20*jy**2 + 11*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 2*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.arccos(-(1/e_per))**4)*f1**2*f2**2 - 18*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(9*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(-27*(5*(-5 + 2*e_per**2)*ex**3 + 4*jx*((-1 + 2*e_per**2)*ey*jy - ez*jz) + ex*(-5*(-5 + 2*e_per**2)*ey**2 + 5*(-1 + 2*e_per**2)*ez**2 + 9*jx**2 - 10*e_per**2*jx**2 - 5*jy**2 + 2*e_per**2*jy**2 + 25*jz**2 - 10*e_per**2*jz**2))*np.pi**5 - 3*(5*(-25 + 4*e_per**2)*ex**3 - 4*jx*(5*ey*(jy + 4*e_per**2*jy) + (5 - 6*e_per**2)*ez*jz) + ex*(-5*(-25 + 4*e_per**2)*ey**2 - 25*(1 + 4*e_per**2)*ez**2 + 45*jx**2 + 76*e_per**2*jx**2 - 25*jy**2 + 4*e_per**2*jy**2 + 125*jz**2 - 20*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 2*(5*(-5 + 8*e_per**2)*ex**3 - 4*jx*(ey*(jy + 8*e_per**2*jy) + (1 - 4*e_per**2)*ez*jz) + ex*(-5*(-5 + 8*e_per**2)*ey**2 - 5*(1 + 8*e_per**2)*ez**2 + 9*jx**2 + 24*e_per**2*jx**2 - 5*jy**2 + 8*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f1**2*f2**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*(9*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2)) + 3*e_per*(5*ex*ez**2 + 2*ey*jx*jy - 2*ex*jy**2 - ez*jx*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per))*f1**2*f2**2*f3 - 12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*(-8*(2*jx*((7 - 4*e_per**2)*ey*jy + (-2 + 5*e_per**2)*ez*jz) + ex*(25*(-2 + e_per**2)*ey**2 - 5*ez**2 + 4*jy**2 - e_per**2*jy**2 + 25*jz**2 - 10*e_per**2*jz**2))*np.pi**4 + 2*(2*jx*(-5*(7 + 2*e_per**2)*ey*jy + (10 + 53*e_per**2)*ez*jz) + ex*(25*(10 + e_per**2)*ey**2 - 5*(-5 + 6*e_per**2)*ez**2 - 20*jy**2 + 11*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 3*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.arccos(-(1/e_per))**4)*f1**2*f3**2 - 12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(-8*(5*(-5 + 2*e_per**2)*ex**3 + 4*jx*((-1 + 2*e_per**2)*ey*jy - ez*jz) + ex*(-5*(-5 + 2*e_per**2)*ey**2 + 5*(-1 + 2*e_per**2)*ez**2 + 9*jx**2 - 10*e_per**2*jx**2 - 5*jy**2 + 2*e_per**2*jy**2 + 25*jz**2 - 10*e_per**2*jz**2))*np.pi**5 + 2*(-5*(-25 + 4*e_per**2)*ex**3 + 4*jx*(5*ey*(jy + 4*e_per**2*jy) + (5 - 6*e_per**2)*ez*jz) + ex*(5*(-25 + 4*e_per**2)*ey**2 + 25*(1 + 4*e_per**2)*ez**2 - 45*jx**2 - 76*e_per**2*jx**2 + 25*jy**2 - 4*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 3*(5*(-5 + 8*e_per**2)*ex**3 - 4*jx*(ey*(jy + 8*e_per**2*jy) + (1 - 4*e_per**2)*ez*jz) + ex*(-5*(-5 + 8*e_per**2)*ey**2 - 5*(1 + 8*e_per**2)*ez**2 + 9*jx**2 + 24*e_per**2*jx**2 - 5*jy**2 + 8*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f1**2*f3**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*(4*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2)) + 3*e_per*(5*ex*ez**2 + 2*ey*jx*jy - 2*ex*jy**2 - ez*jx*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per))*f1**2*f2*f3**2 - 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*((2*jx*((-7 + 4*e_per**2)*ey*jy + (2 - 5*e_per**2)*ez*jz) + ex*(-25*(-2 + e_per**2)*ey**2 + 5*ez**2 - 4*jy**2 + e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi**4 + (2*jx*(-5*(7 + 2*e_per**2)*ey*jy + (10 + 53*e_per**2)*ez*jz) + ex*(25*(10 + e_per**2)*ey**2 - 5*(-5 + 6*e_per**2)*ez**2 - 20*jy**2 + 11*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 6*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.arccos(-(1/e_per))**4)*f2**2*f3**2 - 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*((-5*(-5 + 2*e_per**2)*ex**3 + 4*jx*(ey*(jy - 2*e_per**2*jy) + ez*jz) + ex*(5*(-5 + 2*e_per**2)*ey**2 + (5 - 10*e_per**2)*ez**2 - 9*jx**2 + 10*e_per**2*jx**2 + 5*jy**2 - 2*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi**5 + (-5*(-25 + 4*e_per**2)*ex**3 + 4*jx*(5*ey*(jy + 4*e_per**2*jy) + (5 - 6*e_per**2)*ez*jz) + ex*(5*(-25 + 4*e_per**2)*ey**2 + 25*(1 + 4*e_per**2)*ez**2 - 45*jx**2 - 76*e_per**2*jx**2 + 25*jy**2 - 4*e_per**2*jy**2 - 125*jz**2 + 20*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 6*(5*(-5 + 8*e_per**2)*ex**3 - 4*jx*(ey*(jy + 8*e_per**2*jy) + (1 - 4*e_per**2)*ez*jz) + ex*(-5*(-5 + 8*e_per**2)*ey**2 - 5*(1 + 8*e_per**2)*ez**2 + 9*jx**2 + 24*e_per**2*jx**2 - 5*jy**2 + 8*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f2**2*f3**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*((-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-2*jx*((-7 + 4*e_per**2)*ey*jy + 2*(1 + 11*e_per**2)*ez*jz) + ex*(50*(-1 + e_per**2)*ey**2 + 5*(-1 + 4*e_per**2)*ez**2 + 4*jy**2 - 10*e_per**2*jy**2 + 25*jz**2 - 40*e_per**2*jz**2)) + 3*e_per*(5*ex*ez**2 + 2*ey*jx*jy - 2*ex*jy**2 - ez*jx*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per))*f1*f2**2*f3**2))/np.pi + (ex*(18*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(9*(5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*(-27*(25*(-2 + e_per**2)*ex**2*ey + 2*(7 - 3*e_per**2)*ex*jx*jy - 2*(2 + 3*e_per**2)*ez*jy*jz + ey*(5*(-1 + e_per**2)*ez**2 + (4 - 3*e_per**2)*jx**2 + 5*(5 - 3*e_per**2)*jz**2))*np.pi**4 + 3*(25*(10 + e_per**2)*ex**2*ey + 2*(-35 + 3*e_per**2)*ex*jx*jy + 2*(10 - 51*e_per**2)*ez*jy*jz + 5*ey*((5 + 7*e_per**2)*ez**2 - (4 + 3*e_per**2)*jx**2 - (25 + 9*e_per**2)*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 2*(50*(-1 + e_per**2)*ex**2*ey + 2*(7 - 10*e_per**2)*ex*jx*jy + 4*(-1 + 13*e_per**2)*ez*jy*jz + ey*(-5*(1 + 2*e_per**2)*ez**2 + 2*(2 + e_per**2)*jx**2 + 5*(5 - 2*e_per**2)*jz**2))*np.arccos(-(1/e_per))**4)*f1**2*f2**2 - 18*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(9*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(-27*(5*(-5 + 3*e_per**2)*ex**2*ey - 5*(-5 + 3*e_per**2)*ey**3 + 4*(1 + e_per**2)*ex*jx*jy - 4*(-1 + e_per**2)*ez*jy*jz + ey*(5*(1 + e_per**2)*ez**2 + (5 - 3*e_per**2)*jx**2 - 9*jy**2 - e_per**2*jy**2 - 25*jz**2 + 15*e_per**2*jz**2))*np.pi**5 + 3*(5*(25 + 9*e_per**2)*ex**2*ey - 5*(25 + 9*e_per**2)*ey**3 + 4*(-5 + 19*e_per**2)*ex*jx*jy - 4*(5 + 7*e_per**2)*ez*jy*jz + ey*(5*(-5 + 19*e_per**2)*ez**2 - (25 + 9*e_per**2)*jx**2 + 45*jy**2 - 67*e_per**2*jy**2 + 125*jz**2 + 45*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 2*(5*(-5 + 2*e_per**2)*ex**2*ey - 5*(-5 + 2*e_per**2)*ey**3 + 4*(1 - 10*e_per**2)*ex*jx*jy + 4*(1 + 2*e_per**2)*ez*jy*jz + ey*((5 - 50*e_per**2)*ez**2 + (5 - 2*e_per**2)*jx**2 - 9*jy**2 + 42*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f1**2*f2**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*(9*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-50*(-1 + e_per**2)*ex**2*ey + 2*(-7 + 10*e_per**2)*ex*jx*jy + 4*(1 - 13*e_per**2)*ez*jy*jz + ey*(5*(1 + 2*e_per**2)*ez**2 - 2*(2 + e_per**2)*jx**2 + 5*(-5 + 2*e_per**2)*jz**2)) + 3*e_per*(5*ey*ez**2 - 2*ey*jx**2 + 2*ex*jx*jy - ez*jy*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per))*f1**2*f2**2*f3 + 12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*(-8*(25*(-2 + e_per**2)*ex**2*ey + 2*(7 - 3*e_per**2)*ex*jx*jy - 2*(2 + 3*e_per**2)*ez*jy*jz + ey*(5*(-1 + e_per**2)*ez**2 + (4 - 3*e_per**2)*jx**2 + 5*(5 - 3*e_per**2)*jz**2))*np.pi**4 + 2*(25*(10 + e_per**2)*ex**2*ey + 2*(-35 + 3*e_per**2)*ex*jx*jy + 2*(10 - 51*e_per**2)*ez*jy*jz + 5*ey*((5 + 7*e_per**2)*ez**2 - (4 + 3*e_per**2)*jx**2 - (25 + 9*e_per**2)*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 3*(50*(-1 + e_per**2)*ex**2*ey + 2*(7 - 10*e_per**2)*ex*jx*jy + 4*(-1 + 13*e_per**2)*ez*jy*jz + ey*(-5*(1 + 2*e_per**2)*ez**2 + 2*(2 + e_per**2)*jx**2 + 5*(5 - 2*e_per**2)*jz**2))*np.arccos(-(1/e_per))**4)*f1**2*f3**2 - 12*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(-8*(5*(-5 + 3*e_per**2)*ex**2*ey - 5*(-5 + 3*e_per**2)*ey**3 + 4*(1 + e_per**2)*ex*jx*jy - 4*(-1 + e_per**2)*ez*jy*jz + ey*(5*(1 + e_per**2)*ez**2 + (5 - 3*e_per**2)*jx**2 - 9*jy**2 - e_per**2*jy**2 - 25*jz**2 + 15*e_per**2*jz**2))*np.pi**5 + 2*(5*(25 + 9*e_per**2)*ex**2*ey - 5*(25 + 9*e_per**2)*ey**3 + 4*(-5 + 19*e_per**2)*ex*jx*jy - 4*(5 + 7*e_per**2)*ez*jy*jz + ey*(5*(-5 + 19*e_per**2)*ez**2 - (25 + 9*e_per**2)*jx**2 + 45*jy**2 - 67*e_per**2*jy**2 + 125*jz**2 + 45*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 3*(5*(-5 + 2*e_per**2)*ex**2*ey - 5*(-5 + 2*e_per**2)*ey**3 + 4*(1 - 10*e_per**2)*ex*jx*jy + 4*(1 + 2*e_per**2)*ez*jy*jz + ey*((5 - 50*e_per**2)*ez**2 + (5 - 2*e_per**2)*jx**2 - 9*jy**2 + 42*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f1**2*f3**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*(4*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-50*(-1 + e_per**2)*ex**2*ey + 2*(-7 + 10*e_per**2)*ex*jx*jy + 4*(1 - 13*e_per**2)*ez*jy*jz + ey*(5*(1 + 2*e_per**2)*ez**2 - 2*(2 + e_per**2)*jx**2 + 5*(-5 + 2*e_per**2)*jz**2)) + 3*e_per*(5*ey*ez**2 - 2*ey*jx**2 + 2*ex*jx*jy - ez*jy*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per))*f1**2*f2*f3**2 + 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**3 + (-5 + 8*e_per**2)*np.pi*np.arccos(-(1/e_per))**2)*((-25*(-2 + e_per**2)*ex**2*ey + 2*(-7 + 3*e_per**2)*ex*jx*jy + 2*(2 + 3*e_per**2)*ez*jy*jz + ey*(-5*(-1 + e_per**2)*ez**2 + (-4 + 3*e_per**2)*jx**2 + 5*(-5 + 3*e_per**2)*jz**2))*np.pi**4 + (25*(10 + e_per**2)*ex**2*ey + 2*(-35 + 3*e_per**2)*ex*jx*jy + 2*(10 - 51*e_per**2)*ez*jy*jz + 5*ey*((5 + 7*e_per**2)*ez**2 - (4 + 3*e_per**2)*jx**2 - (25 + 9*e_per**2)*jz**2))*np.pi**2*np.arccos(-(1/e_per))**2 + 6*(50*(-1 + e_per**2)*ex**2*ey + 2*(7 - 10*e_per**2)*ex*jx*jy + 4*(-1 + 13*e_per**2)*ez*jy*jz + ey*(-5*(1 + 2*e_per**2)*ez**2 + 2*(2 + e_per**2)*jx**2 + 5*(5 - 2*e_per**2)*jz**2))*np.arccos(-(1/e_per))**4)*f2**2*f3**2 - 6*(-1 + e_per**2)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*((-5*(-5 + 3*e_per**2)*ex**2*ey + 5*(-5 + 3*e_per**2)*ey**3 - 4*(1 + e_per**2)*ex*jx*jy + 4*(-1 + e_per**2)*ez*jy*jz + ey*(-5*(1 + e_per**2)*ez**2 + (-5 + 3*e_per**2)*jx**2 + 9*jy**2 + e_per**2*jy**2 + 25*jz**2 - 15*e_per**2*jz**2))*np.pi**5 + (5*(25 + 9*e_per**2)*ex**2*ey - 5*(25 + 9*e_per**2)*ey**3 + 4*(-5 + 19*e_per**2)*ex*jx*jy - 4*(5 + 7*e_per**2)*ez*jy*jz + ey*(5*(-5 + 19*e_per**2)*ez**2 - (25 + 9*e_per**2)*jx**2 + 45*jy**2 - 67*e_per**2*jy**2 + 125*jz**2 + 45*e_per**2*jz**2))*np.pi**3*np.arccos(-(1/e_per))**2 + 6*(5*(-5 + 2*e_per**2)*ex**2*ey - 5*(-5 + 2*e_per**2)*ey**3 + 4*(1 - 10*e_per**2)*ex*jx*jy + 4*(1 + 2*e_per**2)*ez*jy*jz + ey*((5 - 50*e_per**2)*ez**2 + (5 - 2*e_per**2)*jx**2 - 9*jy**2 + 42*e_per**2*jy**2 - 25*jz**2 + 10*e_per**2*jz**2))*np.pi*np.arccos(-(1/e_per))**4)*f2**2*f3**2 + np.sqrt(1 - e_per**(-2))*e_per**2*np.pi*((-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(-50*(-1 + e_per**2)*ex**2*ey + 2*(-7 + 10*e_per**2)*ex*jx*jy + 4*(1 - 13*e_per**2)*ez*jy*jz + ey*(5*(1 + 2*e_per**2)*ez**2 - 2*(2 + e_per**2)*jx**2 + 5*(-5 + 2*e_per**2)*jz**2)) + 3*e_per*(5*ey*ez**2 - 2*ey*jx**2 + 2*ex*jx*jy - ez*jy*jz)*np.arccos(-(1/e_per)) - 15*e_per*jz*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per))*f1*f2**2*f3**2))/np.pi - 12*np.sqrt(1 - e_per**(-2))*e_per**2*ez*((-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2*np.arccos(-1.0/e_per)*(2*np.sqrt(1 - e_per**(-2))*((ey*jx - ex*jy)*jz + e_per**2*(5*ex*ey*ez + 3*ez*jx*jy - ey*jx*jz + ex*jy*jz)) + e_per*(6*ez*jx*jy + ey*jx*jz + ex*jy*jz)*np.arccos(-1.0/e_per))*f1*f2*f3*(f2*f3 + f1*(f2 + f3)) - e_per*(-5 + 2*e_per**2)*(10*ex*ey*ez - ey*jx*jz - ex*jy*jz)*np.pi**2*np.arccos(-(1/e_per))*np.arccos(-1.0/e_per)*f1*f2*f3*(f2*f3 + f1*(9*f2 + 4*f3)) - (-5 + 2*e_per**2)*np.pi**2*np.arccos(-1.0/e_per)*(2*np.sqrt(1 - e_per**(-2))*((ey*jx - ex*jy)*jz + e_per**2*(5*ex*ey*ez + 3*ez*jx*jy - ey*jx*jz + ex*jy*jz)) + e_per*(6*ez*jx*jy + ey*jx*jz + ex*jy*jz)*np.arccos(-1.0/e_per))*f1*f2*f3*(f2*f3 + f1*(9*f2 + 4*f3)) + 36*np.sqrt(1 - e_per**(-2))*(-5 + 8*e_per**2)*(4*(ey*jx - ex*jy)*jz + e_per**2*(5*ex*ey*ez + 5*ez*jx*jy - ey*jx*jz + 7*ex*jy*jz))*np.arccos(-(1/e_per))**9*(f2**2*f3**2 + f1**2*(f2**2 + f3**2)) - np.sqrt(1 - e_per**(-2))*(1320*(-(ey*jx*jz) + ex*jy*jz) + 16*e_per**4*(55*ex*ey*ez + 55*ez*jx*jy + 21*ey*jx*jz + 45*ex*jy*jz) - e_per**2*(1225*ex*ey*ez + 1225*ez*jx*jy - 1173*ey*jx*jz + 2643*ex*jy*jz))*np.pi**2*np.arccos(-(1/e_per))**7*(f2**2*f3**2 + f1**2*(9*f2**2 + 4*f3**2)) + 2*np.sqrt(1 - e_per**(-2))*(e_per**4*(85*ex*ey*ez + 85*ez*jx*jy + 111*ey*jx*jz - 9*ex*jy*jz) + 240*(-(ey*jx*jz) + ex*jy*jz) - e_per**2*(175*ex*ey*ez + 175*ez*jx*jy + 141*ey*jx*jz + 69*ex*jy*jz))*np.pi**4*np.arccos(-(1/e_per))**5*(f2**2*f3**2 + f1**2*(81*f2**2 + 16*f3**2)) + np.arccos(-(1/e_per))**3*(e_per*(-5 + 8*e_per**2)*(10*ex*ey*ez - ey*jx*jz - ex*jy*jz)*np.arccos(-1.0/e_per)*f1*f2*f3*(f2*f3 + f1*(f2 + f3)) - np.sqrt(1 - e_per**(-2))*(-5 + 2*e_per**2)*(24*(-(ey*jx) + ex*jy)*jz + e_per**2*(5*ex*ey*ez + 5*ez*jx*jy + 15*ey*jx*jz - 9*ex*jy*jz))*np.pi**6*(f2**2*f3**2 + f1**2*(729*f2**2 + 64*f3**2))))))/(2.*e_per**4*np.sqrt(ex**2 + ey**2 + ez**2)*f1**2*f2**2*f3**2)
             
        Delta_i = -eps_SA*((np.sqrt(1 - e_per**(-2))*(-1 + e_per**2)*(5*ex*ey - jx*jy))/e_per) \
            + eps_SA**2*(-(np.sqrt(1 - e_per**(-2))*(-1 + e_per**2)*(np.sqrt(1 - e_per**(-2))*(10*(1 + 2*e_per**2)*ex*ez*jx + 10*(1 - 4*e_per**2)*ey*ez*jy + 5*(-5 + 8*e_per**2)*ex**2*jz + 5*(-5 + 2*e_per**2)*ey**2*jz + (-1 + 4*e_per**2)*jx**2*jz - (1 + 2*e_per**2)*jy**2*jz) - 3*e_per*(5*ex*ez*jx - 5*ey*ez*jy + (-jx**2 + jy**2)*jz)*np.arccos(-(1/e_per)) + 15*e_per*(3*ex*ez*jx + ex**2*jz - ey*(3*ez*jy + ey*jz))*np.arccos(-1.0/e_per)))/(4.*e_per**2)) \
            + (9*eps_SA**2*np.arccos(-(1/e_per))*np.arccos(-1.0/e_per)*((-2*(-1 + e_per**2)*(5*ey*ez*jx + 5*ex*ez*jy + 5*ex*ey*jz + jx*jy*jz)*np.arccos(-(1/e_per))**3*((5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4))/f1**2 + (np.sqrt(1 - e_per**(-2))*e_per*((-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(2*np.sqrt(1 - e_per**(-2))*e_per*(5*ey*ez*jx + 5*ex*ez*jy + 5*ex*ey*jz + jx*jy*jz) + (-5*ey*ez*jx - 5*ex*ez*jy + 2*jx*jy*jz)*np.arccos(-(1/e_per)) + 5*(3*ey*ez*jx + 3*ex*ez*jy + 2*ex*ey*jz)*np.arccos(-1.0/e_per)))/f1 - (8*(-1 + e_per**2)*(5*ey*ez*jx + 5*ex*ez*jy + 5*ex*ey*jz + jx*jy*jz)*np.arccos(-(1/e_per))**3*(4*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(4*np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4))/f2**2 + (np.sqrt(1 - e_per**(-2))*e_per*(4*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(2*np.sqrt(1 - e_per**(-2))*e_per*(5*ey*ez*jx + 5*ex*ez*jy + 5*ex*ey*jz + jx*jy*jz) + (-5*ey*ez*jx - 5*ex*ez*jy + 2*jx*jy*jz)*np.arccos(-(1/e_per)) + 5*(3*ey*ez*jx + 3*ex*ez*jy + 2*ex*ey*jz)*np.arccos(-1.0/e_per)))/f2 - (18*(-1 + e_per**2)*(5*ey*ez*jx + 5*ex*ez*jy + 5*ex*ey*jz + jx*jy*jz)*np.arccos(-(1/e_per))**3*(9*(5 - 2*e_per**2)*np.pi**2 + (-5 + 8*e_per**2)*np.arccos(-(1/e_per))**2)*(9*np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4))/f3**2 + (np.sqrt(1 - e_per**(-2))*e_per*(9*(-5 + 2*e_per**2)*np.pi**2 + (5 - 8*e_per**2)*np.arccos(-(1/e_per))**2)*np.arccos(-1.0/e_per)*(2*np.sqrt(1 - e_per**(-2))*e_per*(5*ey*ez*jx + 5*ex*ez*jy + 5*ex*ey*jz + jx*jy*jz) + (-5*ey*ez*jx - 5*ex*ez*jy + 2*jx*jy*jz)*np.arccos(-(1/e_per)) + 5*(3*ey*ez*jx + 3*ex*ez*jy + 2*ex*ey*jz)*np.arccos(-1.0/e_per)))/f3))/e_per**2
        Delta_i *= -(1.0/(np.sqrt(jx**2+jy**2)))

        Delta_i = -ArcCos(jz/Sqrt(jx**2 + jy**2 + jz**2)) + ArcCos((jz - (Sqrt(1 - eper**(-2))*(-1 + eper**2)*(5*ex*ey - jx*jy)*eps_SA)/eper - (Sqrt(1 - eper**(-2))*(-1 + eper**2)*eps_SA**2*(Sqrt(1 - eper**(-2))*(10*(1 + 2*eper**2)*ex*ez*jx + 10*(1 - 4*eper**2)*ey*ez*jy + 5*(-5 + 8*eper**2)*ex**2*jz + 5*(-5 + 2*eper**2)*ey**2*jz + (-1 + 4*eper**2)*jx**2*jz - (1 + 2*eper**2)*jy**2*jz) - 3*eper*(5*ex*ez*jx - 5*ey*ez*jy + (-jx**2 + jy**2)*jz)*ArcCos(-(1/eper)) + 15*eper*(3*ex*ez*jx + ex**2*jz - ey*(3*ez*jy + ey*jz))*ArcCos(-1.0/eper)))/(4.*eper**2))/Sqrt((jz - (Sqrt(1 - eper**(-2))*(-1 + eper**2)*(5*ex*ey - jx*jy)*eps_SA)/eper - (Sqrt(1 - eper**(-2))*(-1 + eper**2)*eps_SA**2*(Sqrt(1 - eper**(-2))*(10*(1 + 2*eper**2)*ex*ez*jx + 10*(1 - 4*eper**2)*ey*ez*jy + 5*(-5 + 8*eper**2)*ex**2*jz + 5*(-5 + 2*eper**2)*ey**2*jz + (-1 + 4*eper**2)*jx**2*jz - (1 + 2*eper**2)*jy**2*jz) - 3*eper*(5*ex*ez*jx - 5*ey*ez*jy + (-jx**2 + jy**2)*jz)*ArcCos(-(1/eper)) + 15*eper*(3*ex*ez*jx + ex**2*jz - ey*(3*ez*jy + ey*jz))*ArcCos(-1.0/eper)))/(4.*eper**2))**2 + (jy + ((5*ex*ez - jx*jz)*eps_SA*(Sqrt(1 - eper**(-2))*(-1 + 4*eper**2) + 3*eper*ArcCos(-(1/eper))))/(2.*eper) - (eps_SA**2*(Sqrt(1 - eper**(-2))*(-1 + 4*eper**2) + 3*eper*ArcCos(-(1/eper)))*(3*eper*(-10*ex*ey*jx + 10*ex**2*jy + jz*(-5*ey*ez + jy*jz))*ArcCos(-(1/eper))*f1*f2*f3 + (Sqrt(1 - eper**(-2))*(-10*(1 + 2*eper**2)*ex*ey*jx + 10*(-2 + 5*eper**2)*ex**2*jy + 5*(-1 + 10*eper**2)*ez**2*jy + 2*(-1 + eper**2)*jx**2*jy - 20*(-1 + eper**2)*ey*ez*jz + (1 + 2*eper**2)*jy*jz**2) + 15*eper*ez*(3*ez*jy + ey*jz)*ArcCos(-1.0/eper))*f1*f2*f3 + 12*Sqrt(1 - eper**(-2))*(-5 + 8*eper**2)*(15*ex**2*jx - 20*ex*(ey*jy + ez*jz) + jx*(5*ey**2 + 5*ez**2 + jx**2 - jy**2 - jz**2))*ArcCos(-(1/eper))**5*(f2*f3 + f1*(f2 + f3)) - 12*Sqrt(1 - eper**(-2))*(-5 + 2*eper**2)*(15*ex**2*jx - 20*ex*(ey*jy + ez*jz) + jx*(5*ey**2 + 5*ez**2 + jx**2 - jy**2 - jz**2))*Pi**2*ArcCos(-(1/eper))**3*(f2*f3 + f1*(9*f2 + 4*f3))))/(8.*eper**2*f1*f2*f3))**2 + (jx - ((5*ey*ez - jy*jz)*eps_SA*(Sqrt(1 - eper**(-2))*(1 + 2*eper**2) + 3*eper*ArcCos(-(1/eper))))/(2.*eper) - (eps_SA**2*(Sqrt(1 - eper**(-2))*(1 + 2*eper**2) + 3*eper*ArcCos(-(1/eper)))*(3*eper*(10*ey**2*jx - 10*ex*ey*jy + jz*(-5*ex*ez + jx*jz))*ArcCos(-(1/eper))*f1*f2*f3 + (Sqrt(1 - eper**(-2))*(10*(2 + eper**2)*ey**2*jx + 5*ez**2*(jx + 8*eper**2*jx) + 10*(1 - 4*eper**2)*ex*ey*jy + 20*(-1 + eper**2)*ex*ez*jz - jx*(2*(-1 + eper**2)*jy**2 + (1 - 4*eper**2)*jz**2)) + 15*eper*ez*(3*ez*jx + ex*jz)*ArcCos(-1.0/eper))*f1*f2*f3 + 12*Sqrt(1 - eper**(-2))*(-5 + 8*eper**2)*(-20*ex*ey*jx + 5*ex**2*jy + 15*ey**2*jy - 20*ey*ez*jz + jy*(5*ez**2 - jx**2 + jy**2 - jz**2))*ArcCos(-(1/eper))**5*(f2*f3 + f1*(f2 + f3)) - 12*Sqrt(1 - eper**(-2))*(-5 + 2*eper**2)*(-20*ex*ey*jx + 5*ex**2*jy + 15*ey**2*jy - 20*ey*ez*jz + jy*(5*ez**2 - jx**2 + jy**2 - jz**2))*Pi**2*ArcCos(-(1/eper))**3*(f2*f3 + f1*(9*f2 + 4*f3))))/(8.*eper**2*f1*f2*f3))**2))
        
    return Delta_e_CDA,Delta_i


def plot_function(args,data):
    a = args.a
    e = args.e
    thetas = data["thetas"]
    e_sol = data["e_sol"]
    j_sol = data["j_sol"]
    i_sol = data["i_sol"]
    Delta_e = data["Delta_e"]
    Delta_i = data["Delta_i"]
    
    print 'Delta_e',Delta_e
    fontsize=args.fontsize
    labelsize=args.labelsize
    
    fig=pyplot.figure(figsize=(8,10))
    plot1=fig.add_subplot(2,1,1,yscale="linear")
    plot2=fig.add_subplot(2,1,2,yscale="linear")
    
    plot1.plot(thetas*180.0/np.pi,e_sol,color='k')

    plot2.plot(thetas*180.0/np.pi,np.array(i_sol)*180.0/np.pi,color='k')

    plots = [plot1,plot2]
    labels = [r"$e$","$i/\mathrm{deg}$"]
    for index,plot in enumerate(plots):
        if index==1:
            plot.set_xlabel(r"$\theta/\mathrm{deg}$",fontsize=fontsize)
        plot.set_ylabel(labels[index],fontsize=fontsize)
                
        plot.tick_params(axis='both', which ='major', labelsize = labelsize)

        if index in [0]:
            plot.set_xticklabels([])

    fig.subplots_adjust(hspace=0.0,wspace=0.0)
    filename = os.path.dirname(os.path.realpath(__file__)) + '/figs/elements_' + str(args.name) + '_e_per_' + str(args.e_per) + '_Q_div_a_' + str(int(args.Q/args.a)) + '.pdf'
    fig.savefig(filename,dpi=200)

def plot_function_fourier(args,data):
    a = args.a
    e = args.e
    e_per = args.e_per
    m = args.m
    M_per = args.M_per
    Q = args.Q
    thetas = data["thetas"]
    e_sol = data["e_sol"]
    j_sol = data["j_sol"]
    i_sol = data["i_sol"]
    Delta_e = data["Delta_e"]
    Delta_i = data["Delta_i"]

    eps_SA = compute_eps_SA(m,M_per,a/Q,e_per)

    i = args.i
    omega = args.omega
    Omega = args.Omega
    
    ex0,ey0,ez0,jx0,jy0,jz0 = orbital_elements_to_orbital_vectors(e,i,omega,Omega)

    fig=pyplot.figure(figsize=(8.5,10))
    plot1=fig.add_subplot(2,1,1,yscale="linear")
    plot2=fig.add_subplot(2,1,2,yscale="linear")

    lmaxs = [1,2,3]

    linewidths=[1.5,1.5,1.5]
    linestyles=['dotted','dashed','solid']

    fontsize=args.fontsize
    labelsize=args.labelsize
    
    for index_lmax,lmax in enumerate(lmaxs):

        e_f = []
        i_f = []
        ex_f = []
        jz_f = []
        for i,theta in enumerate(thetas):
            ex,ey,ez,jx,jy,jz = fourier_series_ej(lmax,theta,eps_SA,e_per,ex0,ey0,ez0,jx0,jy0,jz0)
            e = np.sqrt(ex**2+ey**2+ez**2)
            e_f.append(e)
            ex_f.append(ex)
            jz_f.append(jz)
            j = np.sqrt(1.0-e**2)
            i = np.arccos(jz/j)
            i_f.append(i)
        
        print 'Delta_e',Delta_e
        if index_lmax==0:
            plot1.plot(thetas*180.0/np.pi,e_sol,color='k',label='$\mathrm{SA}$',linestyle='dashed',zorder=10,linewidth=2)
            plot2.plot(thetas*180.0/np.pi,np.array(i_sol)*180.0/np.pi,color='k',linestyle='dashed',zorder=10,linewidth=2)
        plot1.plot(thetas*180.0/np.pi,e_f,color='r',label='$\mathrm{Fourier}; \, l_\mathrm{max}=%s$'%lmax,linestyle=linestyles[index_lmax],linewidth=linewidths[index_lmax])
        plot2.plot(thetas*180.0/np.pi,np.array(i_f)*180.0/np.pi,color='r',linestyle=linestyles[index_lmax],linewidth=linewidths[index_lmax])
        

    plots = [plot1,plot2]
    labels = [r"$e$","$i/\mathrm{deg}$"]
    for index,plot in enumerate(plots):
        if index==1:
            plot.set_xlabel(r"$\theta/\mathrm{deg}$",fontsize=fontsize)
        plot.set_ylabel(labels[index],fontsize=fontsize)
                
        plot.tick_params(axis='both', which ='major', labelsize = labelsize)

        if index in [0]:
            plot.set_xticklabels([])

    handles,labels = plot1.get_legend_handles_labels()
    plot1.legend(handles,labels,loc="upper left",fontsize=0.8*fontsize)
    
    plot1.set_title("$Q/a=%s; \, E = %s; \,e=%s;\,i=%s\,\mathrm{deg};\, \omega=%s\,\mathrm{deg};\, \Omega=%s\,\mathrm{deg}$"%(round(args.Q/args.a,1),args.e_per,args.e,round(args.i*180.0/np.pi,1),round(args.omega*180.0/np.pi,1),round(args.Omega*180.0/np.pi,1)),fontsize=0.7*fontsize)
    
    fig.subplots_adjust(hspace=0.0,wspace=0.0)
    filename = os.path.dirname(os.path.realpath(__file__)) + '/figs/fourier_' + 'Q_div_a_' + str(int(args.Q/args.a)) + '_' + str(args.name) + '.pdf'
    fig.savefig(filename,dpi=200)
    

    
def plot_function_series(args,data_series):
    Q_points = data_series["Q_points"]
    Delta_es = data_series["Delta_es"]
    Delta_is = data_series["Delta_is"]
    do_nbody = data_series["do_nbody"]
    nbody_Delta_es = data_series["nbody_Delta_es"]
    nbody_Delta_is = data_series["nbody_Delta_is"]

    m = args.m
    M_per = args.M_per
    m1 = args.m1
    m2 = args.m2
    e_per = args.e_per
    a = args.a
    e = args.e
    
    Q_div_a_points = Q_points/a

    i = args.i
    omega = args.omega
    Omega = args.Omega
    
    ex,ey,ez,jx,jy,jz = orbital_elements_to_orbital_vectors(e,i,omega,Omega)
    
    Delta_es_DA,Delta_is_DA = [],[]
    Delta_es_CDA,Delta_is_CDA = [],[]
    
    plot_Q_div_a_points = pow(10.0,np.linspace(np.log10(np.amin(Q_div_a_points)),np.log10(np.amax(Q_div_a_points)),1000))
    
    for index,Q_div_a in enumerate(plot_Q_div_a_points):
        a_div_Q = 1.0/Q_div_a
        eps_SA = compute_eps_SA(m,M_per,a_div_Q,e_per)
        eps_oct = compute_eps_oct(m1,m2,m,a_div_Q,e_per)
        
        Delta_e_DA,Delta_i_DA = compute_DA_prediction(args,eps_SA,e_per,ex,ey,ez,jx,jy,jz)
        Delta_e_CDA,Delta_i_CDA = compute_CDA_prediction(args,eps_SA,e_per,ex,ey,ez,jx,jy,jz)

        if args.include_octupole_terms == True:
            Delta_e_oct,Delta_i_oct = compute_DA_prediction_oct(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz)
            Delta_e_CDA += Delta_e_oct

        Delta_es_DA.append(Delta_e_DA)
        Delta_es_CDA.append(Delta_e_CDA)

        Delta_is_DA.append(Delta_i_DA)
        Delta_is_CDA.append(Delta_i_CDA)
        
        g1 = Delta_e_DA/eps_SA
        g2 = (Delta_e_CDA-Delta_e_DA)/(eps_SA**2)
    
    a_div_Q_crit = pow( (-g1/g2)*(np.sqrt(m*(m+M_per))/M_per),2.0/3.0)*(1.0+e_per)
    Q_div_a_crit = 1.0/a_div_Q_crit

    Q_div_a_crit2 = Q_div_a_crit*pow(0.5,-2.0/3.0)


    Q_div_a_crit_R_unity = pow(0.5,-2.0/3.0)*pow( (1.0 + M_per/m)*(1.0 + e_per), 1.0/3.0)

    Delta_es_DA = np.array(Delta_es_DA)
    Delta_es_CDA = np.array(Delta_es_CDA)
    Delta_is_DA = np.array(Delta_is_DA)
    Delta_is_CDA = np.array(Delta_is_CDA)

    fontsize=args.fontsize
    labelsize=args.labelsize
    
    fig=pyplot.figure(figsize=(8,10))
    plot1=fig.add_subplot(2,1,1,yscale="log",xscale="log")
    plot2=fig.add_subplot(2,1,2,yscale="linear",xscale="log")
    fig_c=pyplot.figure(figsize=(8,6))
    plot_c=fig_c.add_subplot(1,1,1,yscale="log",xscale="log")
    
    s_nbody=50
    s=10
    plots = [plot1,plot_c]
    for plot in plots:
        plot.axvline(x=Q_div_a_crit,color='k',linestyle='dotted')
        plot.axvline(x=Q_div_a_crit2,color='k',linestyle='dotted')
        plot.axvline(x=Q_div_a_crit_R_unity,color='r',linestyle='dotted')
    
        indices_pos = [i for i in range(len(Delta_es)) if Delta_es[i] >= 0.0]
        indices_neg = [i for i in range(len(Delta_es)) if Delta_es[i] < 0.0]
        plot.scatter(Q_div_a_points[indices_pos],Delta_es[indices_pos],color='b',s=s, facecolors='none',label="$\mathrm{SA}$")
        plot.scatter(Q_div_a_points[indices_neg],-Delta_es[indices_neg],color='r',s=s, facecolors='none')

        if do_nbody == True:
            s=s_nbody
            indices_pos = [i for i in range(len(nbody_Delta_es)) if nbody_Delta_es[i] >= 0.0]
            indices_neg = [i for i in range(len(nbody_Delta_es)) if nbody_Delta_es[i] < 0.0]
            plot.scatter(Q_div_a_points[indices_pos],nbody_Delta_es[indices_pos],color='b',s=s, facecolors='none',label="$\mathrm{3-body}$",marker='*')
            plot.scatter(Q_div_a_points[indices_neg],-nbody_Delta_es[indices_neg],color='r',s=s, facecolors='none',marker='*')

        w=2.0
        plot.axhline(y = 1.0 - e,color='k',linestyle='dotted',linewidth=w,label="$1-e$")
        plot.plot(plot_Q_div_a_points,Delta_es_DA,color='k',linestyle='dashed',linewidth=w,label="$\mathrm{FO}$")

        indices_pos = [i for i in range(len(Delta_es_CDA)) if Delta_es_CDA[i] >= 0.0]
        indices_neg = [i for i in range(len(Delta_es_CDA)) if Delta_es_CDA[i] < 0.0]
        plot.plot(plot_Q_div_a_points[indices_pos],Delta_es_CDA[indices_pos],color='b',linestyle='solid',linewidth=w,label="$\mathrm{SO}$")
        plot.plot(plot_Q_div_a_points[indices_neg],-Delta_es_CDA[indices_neg],color='r',linestyle='solid',linewidth=w)    

    if do_nbody == True:
        s=s_nbody
        indices_pos = [i for i in range(len(nbody_Delta_is)) if nbody_Delta_is[i] >= 0.0]
        indices_neg = [i for i in range(len(nbody_Delta_is)) if nbody_Delta_is[i] < 0.0]
        plot2.scatter(Q_div_a_points[indices_pos],nbody_Delta_is[indices_pos]*180.0/np.pi,color='b',s=s, facecolors='none',marker='*')
        plot2.scatter(Q_div_a_points[indices_neg],-nbody_Delta_is[indices_neg]*180.0/np.pi,color='r',s=s, facecolors='none',marker='*')


    indices_pos = [i for i in range(len(Delta_is)) if Delta_is[i] >= 0.0]
    indices_neg = [i for i in range(len(Delta_is)) if Delta_is[i] < 0.0]
    plot2.scatter(Q_div_a_points[indices_pos],Delta_is[indices_pos]*180.0/np.pi,color='b',s=s, facecolors='none')
    plot2.scatter(Q_div_a_points[indices_neg],-Delta_is[indices_neg]*180.0/np.pi,color='r',s=s, facecolors='none')

    indices_pos = [i for i in range(len(Delta_is_DA)) if Delta_is_DA[i] >= 0.0]
    indices_neg = [i for i in range(len(Delta_is_DA)) if Delta_is_DA[i] < 0.0]
    plot2.plot(plot_Q_div_a_points[indices_pos],Delta_is_DA[indices_pos]*180.0/np.pi,color='b',linestyle='dashed',linewidth=w)
    plot2.plot(plot_Q_div_a_points[indices_neg],-Delta_is_DA[indices_neg]*180.0/np.pi,color='r',linestyle='dashed',linewidth=w)


    indices_pos = [i for i in range(len(Delta_is_CDA)) if Delta_is_CDA[i] >= 0.0]
    indices_neg = [i for i in range(len(Delta_is_CDA)) if Delta_is_CDA[i] < 0.0]
    plot2.plot(plot_Q_div_a_points[indices_pos],Delta_is_CDA[indices_pos]*180.0/np.pi,color='b',linestyle='solid',linewidth=w)
    plot2.plot(plot_Q_div_a_points[indices_neg],-Delta_is_CDA[indices_neg]*180.0/np.pi,color='r',linestyle='solid',linewidth=w)
    
    plot1.set_ylim(1.0e-5,1.0e-1)
    plot_c.set_ylim(1.0e-5,1.0e-1)
    plots = [plot1,plot2,plot_c]
    for plot in plots:
        plot.set_xlim(0.95*plot_Q_div_a_points[0],1.05*plot_Q_div_a_points[-1])
    
    plots = [plot1,plot2]
    labels = [r"$\Delta e$",r"$\Delta i/\mathrm{deg}$"]
    for index,plot in enumerate(plots):
        if index==1:
            plot.set_xlabel(r"$Q/a$",fontsize=fontsize)
        plot.set_ylabel(labels[index],fontsize=fontsize)
                
        plot.tick_params(axis='both', which ='major', labelsize = labelsize)

        if index in [0]:
            plot.set_xticklabels([])

    plot_c.set_xlabel(r"$Q/a$",fontsize=fontsize)
    plot_c.set_ylabel(labels[0],fontsize=fontsize)
    plot_c.tick_params(axis='both', which ='major', labelsize = labelsize)

    for plot in [plot1,plot_c]:
        handles,labels = plot.get_legend_handles_labels()
        plot.legend(handles,labels,loc="upper right",fontsize=0.7*fontsize)

        plot.set_title("$E = %s; q = %s; \, \,e=%s;\,i=%s\,\mathrm{deg};\, \omega=%s\,\mathrm{deg};\, \Omega=%s\,\mathrm{deg}$"%(args.e_per,round(args.m1/args.m2,1),args.e,round(args.i*180.0/np.pi,1),round(args.omega*180.0/np.pi,1),round(args.Omega*180.0/np.pi,1)),fontsize=0.65*fontsize)    
    
    
    fig.subplots_adjust(hspace=0.0,wspace=0.0)
    filename = os.path.dirname(os.path.realpath(__file__)) + '/figs/Delta_es_is_' + str(args.name) + '_e_per_' + str(args.e_per) + '_e_' + str(args.e) + '_q_' + str(args.m1/args.m2) + '_i_' + str(args.i) + '_do_nbody_' + str(do_nbody) + '_Q_min_' + str(args.Q_min) + '_Q_max_' + str(args.Q_max) + '_N_Q_' + str(args.N_Q) + '.pdf'
    fig.savefig(filename,dpi=200)

    fig_c.subplots_adjust(hspace=0.0,wspace=0.0)
    filename = os.path.dirname(os.path.realpath(__file__)) + '/figs/Delta_es_' + str(args.name) + '_e_per_' + str(args.e_per) + '_e_' + str(args.e) + '_q_' + str(args.m1/args.m2)  + '_i_' + str(args.i) + '_do_nbody_' + str(do_nbody) + '_Q_min_' + str(args.Q_min) + '_Q_max_' + str(args.Q_max) + '_N_Q_' + str(args.N_Q) + '.pdf'
    fig_c.savefig(filename,dpi=200)
    

def plot_function_series_detailed(args,data_series):
    Q_points = data_series["Q_points"]
    Delta_es = data_series["Delta_es"]
    Delta_is = data_series["Delta_is"]
    thetas_Q = data_series["thetas_Q"]
    es_Q = data_series["es_Q"]
    is_Q = data_series["is_Q"]    

    m = args.m
    M_per = args.M_per
    m1 = args.m1
    m2 = args.m2
    e_per = args.e_per
    a = args.a
    e = args.e
    
    Q_div_a_points = Q_points/a

    i = args.i
    omega = args.omega
    Omega = args.Omega
    
    ex,ey,ez,jx,jy,jz = orbital_elements_to_orbital_vectors(e,i,omega,Omega)
    
    Delta_es_DA,Delta_is_DA = [],[]
    Delta_es_CDA,Delta_is_CDA = [],[]
    
    plot_Q_div_a_points = pow(10.0,np.linspace(np.log10(np.amin(Q_div_a_points)),np.log10(np.amax(Q_div_a_points)),1000))
    
    fig=pyplot.figure(figsize=(8,10))
    plot1=fig.add_subplot(2,1,1)
    plot2=fig.add_subplot(2,1,2)
    
    N_Q = len(Q_points)
    plot_indices = [int(N_Q/4.0),2*int(N_Q/4.0),3*int(N_Q/4.0)]
    linestyles=['solid','dashed','dotted']

    colors = ['k','k','k']
    linewidths = [1.0,1.5,2.0]
    index_plot=0
    fontsize=args.fontsize
    labelsize=args.labelsize

    for index,Q_div_a in enumerate(Q_div_a_points):
        if index not in plot_indices: continue
        
        a_div_Q = 1.0/Q_div_a
        eps_SA = compute_eps_SA(m,M_per,a_div_Q,e_per)
        eps_oct = compute_eps_oct(m1,m2,m,a_div_Q,e_per)
        
        thetas = thetas_Q[index]
        es = es_Q[index]
        incls = is_Q[index]

        eps_SA = compute_eps_SA(m,M_per,a_div_Q,e_per)
        
        Delta_e_DA,Delta_i_DA = compute_DA_prediction(args,eps_SA,e_per,ex,ey,ez,jx,jy,jz)

        label = r"$Q/a\simeq%s$"%(round(Q_div_a,1))
        color=colors[index_plot]
        linewidth=linewidths[index_plot]
        linestyle=linestyles[index_plot]
        plot1.plot(thetas*180.0/np.pi,es,color=color,linestyle=linestyle,linewidth=linewidth,label=label)
        plot2.plot(thetas*180.0/np.pi,np.array(incls)*180.0/np.pi,color=color,linestyle=linestyle,linewidth=linewidth,label=label)

        plot1.axhline(y=e+Delta_e_DA,linestyle=linestyle,linewidth=linewidth,color='r')
        
        index_plot+=1
        
    
    plots=[plot1,plot2]
    labels = [r"$e$",r"$i/\mathrm{deg}$"]
    for index,plot in enumerate(plots):
        if index==1:
            plot.set_xlabel(r"$\theta/\mathrm{deg}$",fontsize=fontsize)
        plot.set_ylabel(labels[index],fontsize=fontsize)
                
        plot.tick_params(axis='both', which ='major', labelsize = labelsize)

        if index in [0]:
            plot.set_xticklabels([])


    handles,labels = plot1.get_legend_handles_labels()
    plot1.legend(handles,labels,loc="upper left",fontsize=0.6*fontsize)

    plot1.set_title("$E = %s; q = %s; \, \,e=%s;\,i=%s\,\mathrm{deg};\, \omega=%s\,\mathrm{deg};\, \Omega=%s\,\mathrm{deg}$"%(args.e_per,round(args.m1/args.m2,1),args.e,round(args.i*180.0/np.pi,1),round(args.omega*180.0/np.pi,1),round(args.Omega*180.0/np.pi,1)),fontsize=0.65*fontsize)    
    
    
    fig.subplots_adjust(hspace=0.0,wspace=0.0)
    filename = os.path.dirname(os.path.realpath(__file__)) + '/figs/elements_time_series_' + str(args.name) + '_e_per_' + str(args.e_per) + '_e_' + str(args.e) + '_q_' + str(args.m1/args.m2) + '_Q_min_' + str(args.Q_min) + '_Q_max_' + str(args.Q_max) + '_N_Q_' + str(args.N_Q) + '.pdf'
    fig.savefig(filename,dpi=200)

    

        
if __name__ == '__main__':
    args = parse_arguments()



    if args.verbose==True:
        print 'arguments:'
        from pprint import pprint
        pprint(vars(args))

    if args.plot>0:
        if HAS_MATPLOTLIB == False:
            print 'Error importing Matplotlib -- choose --plot 0'
            exit(-1)

        if args.plot_fancy == True:
            pyplot.rc('text',usetex=True)
            pyplot.rc('legend',fancybox=True)


    if args.calc == True:
        if args.mode in [1,2]:
            data = integrate(args)
        elif args.mode in [3,4]:
            data_series = integrate_series(args)

    if args.plot == True:
        if args.mode in [1,2]:
            filename = args.data_filename
            with open(filename, 'rb') as handle:
                data = pickle.load(handle)
            
            if args.mode==1:
                plot_function(args,data)
            elif args.mode==2:
                plot_function_fourier(args,data)
                
        elif args.mode in [3,4]:
            filename = args.series_data_filename
            print 'filename',filename
            with open(filename, 'rb') as handle:
                data_series = pickle.load(handle)
            
            if args.mode==3:
                plot_function_series(args,data_series)
            if args.mode==4:
                plot_function_series_detailed(args,data_series)

        else:
            'Incorrect plot id'
            exit(-1)


    if args.plot>0 and args.show == True:
        pyplot.show()
