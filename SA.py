"""
Script to numerically integrate the equations (EOM) for a binary perturbed by a passing body on a hyperbolic orbit.
The EOM are averaged over the binary orbit, but not over the perturber's orbit. 
Valid to the lowest expansion order in r_bin/r_per (i.e., second order or quadrupole order).

Adrian Hamers
March 2019
"""

import argparse
import numpy as np

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

    parser.add_argument("--calc",                           type=int,       dest="calc",                        default=1,              help="calc")
    parser.add_argument("--name",                           type=str,       dest="name",                        default="test01",       help="name")
    parser.add_argument("--M",                              type=float,     dest="M",                           default=2.0,            help="Binary mass")
    parser.add_argument("--M_per",                          type=float,     dest="M_per",                       default=1.0,            help="Perturber mass")
    parser.add_argument("--e_per",                          type=float,     dest="e_per",                       default=1.0+1.0e-10,    help="Perturber eccentricity")
    parser.add_argument("--Q",                              type=float,     dest="Q",                           default=4.0,            help="Perturber periapsis distance (same units as a)")
    parser.add_argument("--Q_min",                          type=float,     dest="Q_min",                       default=2.0,            help="Minimum perturber periapsis distance (in case of series)")
    parser.add_argument("--Q_max",                          type=float,     dest="Q_max",                       default=50.0,           help="Maximum perturber periapsis distance (in case of series)")
    parser.add_argument("--N_Q",                            type=int,       dest="N_Q",                         default=200,            help="Number of systems in series")
    parser.add_argument("--a",                              type=float,     dest="a",                           default=1.0,            help="Binary semimajor axis (same units as Q)")    
    parser.add_argument("--e",                              type=float,     dest="e",                           default=0.999,          help="Binary eccentricity")    
    parser.add_argument("--i",                              type=float,     dest="i",                           default=np.pi/2.0,      help="Binary inclination")    
    parser.add_argument("--omega",                          type=float,     dest="omega",                       default=np.pi/4.0,      help="Binary argument of periapsis")
    parser.add_argument("--Omega",                          type=float,     dest="Omega",                       default=-np.pi/4.0,     help="Binary longitude of the ascending node")
    parser.add_argument("--N_steps",                        type=int,       dest="N_steps",                       default=100,          help="Output steps")
    parser.add_argument("--mxstep",                         type=int,       dest="mxstep",                      default=100000,         help="Maximum number of internal steps taken in the ODE integratino. Increase if ODE integrator gives errors. ")    

    ### boolean arguments ###
    add_bool_arg(parser, 'verbose',                     default=False,          help="verbose terminal output")
    add_bool_arg(parser, 'plot',                        default=True,           help="plotting") 
    add_bool_arg(parser, 'plot_fancy',                  default=False,          help="using LaTeX for plot labels (slower).")
    add_bool_arg(parser, 'include_quadrupole_terms',    default=True,           help="inclusion of secular three-body quadrupole-order terms.")
    add_bool_arg(parser, 'include_octupole_terms',      default=True,           help="inclusion of secular three-body octupole-order terms.")
    add_bool_arg(parser, 'include_1PN_terms',           default=False,          help="inclusion of first post-Newtonian terms")
    
    args = parser.parse_args()
                       
    return args
    

def RHS_function(RHR_vec, theta, *ODE_args):
    ### initialization ###
    eps_SA,e_per,args = ODE_args
    verbose = args.verbose

    ex = RHR_vec[0]
    ey = RHR_vec[1]
    ez = RHR_vec[2]

    jx = RHR_vec[3]
    jy = RHR_vec[4]
    jz = RHR_vec[5]
    
    dex_dtheta = (-3*eps_SA*(1 + e_per*np.cos(theta))*(3*ez*jy + ey*jz + (ez*jy - 5*ey*jz)*np.cos(2*theta) + (-(ez*jx) + 5*ex*jz)*np.sin(2*theta)))/4.
    dey_dtheta = (-3*eps_SA*(1 + e_per*np.cos(theta))*(-2*ez*jx + 2*ex*jz + (ez*jx - 5*ex*jz)*np.cos(theta)**2 + (ez*jy - 5*ey*jz)*np.cos(theta)*np.sin(theta)))/2.
    dez_dtheta = (-3*eps_SA*(1 + e_per*np.cos(theta))*(-(ey*jx) + ex*jy + 2*(ey*jx + ex*jy)*np.cos(2*theta) + (-2*ex*jx + 2*ey*jy)*np.sin(2*theta)))/2.
    djx_dtheta = (3*eps_SA*(1 + e_per*np.cos(theta))*np.sin(theta)*((-5*ex*ez + jx*jz)*np.cos(theta) + (-5*ey*ez + jy*jz)*np.sin(theta)))/2.
    djy_dtheta = (3*eps_SA*np.cos(theta)*(1 + e_per*np.cos(theta))*((5*ex*ez - jx*jz)*np.cos(theta) + (5*ey*ez - jy*jz)*np.sin(theta)))/2.
    djz_dtheta = (3*eps_SA*(1 + e_per*np.cos(theta))*((-10*ex*ey + 2*jx*jy)*np.cos(2*theta) + (5*ex**2 - 5*ey**2 - jx**2 + jy**2)*np.sin(2*theta)))/4.

    RHR_vec_dot = [dex_dtheta,dey_dtheta,dez_dtheta,djx_dtheta,djy_dtheta,djz_dtheta]

    return RHR_vec_dot

def orbital_elements_to_orbital_vectors(e,i,omega,Omega):
    j = np.sqrt(1.0 - e**2)
    ex = e*(np.cos(omega)*np.cos(Omega) - np.cos(i)*np.sin(omega)*np.sin(Omega))
    ey = e*(np.cos(i)*np.cos(Omega)*np.sin(omega) + np.cos(omega)*np.sin(Omega))
    ez = e*np.sin(i)*np.sin(omega)
    jx = j*np.sin(i)*np.sin(Omega)
    jy = -j*np.cos(Omega)*np.sin(i)
    jz = j*np.cos(i)
    return ex,ey,ez,jx,jy,jz
    
def integrate(args):
    if args.verbose==True:
        print 'arguments:'
        from pprint import pprint
        pprint(vars(args))

    if args.plot_fancy == True:
        pyplot.rc('text',usetex=True)
        pyplot.rc('legend',fancybox=True)

    ### initial conditions ###   
    M = args.M
    M_per = args.M_per
    e_per = args.e_per
    Q = args.Q
    
    a = args.a
    e = args.e

    eps_SA = compute_eps_SA(M,M_per,a/Q,e_per)

    i = args.i
    omega = args.omega
    Omega = args.Omega
    
    ex,ey,ez,jx,jy,jz = orbital_elements_to_orbital_vectors(e,i,omega,Omega)
        
    N_steps = args.N_steps
    theta_0 = np.arccos(-1.0/e_per)
    thetas = np.linspace(-theta_0, theta_0, N_steps)

    ODE_args = (eps_SA,e_per,args)

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

    return thetas,ex_sol,ey_sol,ez_sol,jx_sol,jy_sol,jz_sol,e_sol,j_sol,i_sol,Delta_e,Delta_i

def integrate_series(args):
    Q_points = pow(10.0,np.linspace(np.log10(args.Q_min),np.log10(args.Q_max),args.N_Q))
    Delta_es = []
    Delta_is = []
    
    for index_Q,Q in enumerate(Q_points):
        args.Q = Q
        data = integrate(args)
        Delta_e = data[-2]
        Delta_i = data[-1]
        
        Delta_es.append(Delta_e)
        Delta_is.append(Delta_i)

    Delta_es = np.array(Delta_es)
    Delta_is = np.array(Delta_is)
    
    data_series = Q_points,Delta_es,Delta_is
    return data_series
    
def plot_function(args,data):
    a = args.a
    e = args.e
    thetas,ex_sol,ey_sol,ez_sol,jx_sol,jy_sol,jz_sol,e_sol,j_sol,i_sol,Delta_e,Delta_i = data
    
    print 'Delta_e',Delta_e
    fontsize=18
    labelsize=12
    
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

    #handles,labels = plot3.get_legend_handles_labels()
    #plot3.legend(handles,labels,loc="upper right",fontsize=0.8*fontsize)
    
    fig.subplots_adjust(hspace=0.0,wspace=0.0)
    filename = 'elements_' + str(args.name) + '.pdf'
    fig.savefig(filename,dpi=200)
    
    pyplot.show()

def compute_eps_SA(M,M_per,a_div_Q,e_per):
    return (M_per/np.sqrt(M*(M+M_per)))*pow(a_div_Q,3.0/2.0)*pow(1.0 + e_per,-3.0/2.0)
    
def compute_DA_prediction(eps_SA,e_per,ex,ey,ez,jx,jy,jz):
    
    Delta_e = (5*eps_SA*(np.sqrt(1 - e_per**(-2))*((1 + 2*e_per**2)*ey*ez*jx + (1 - 4*e_per**2)*ex*ez*jy + 2*(-1 + e_per**2)*ex*ey*jz) + 3*e_per*ez*(ey*jx - ex*jy)*np.arccos(-1.0/e_per)))/(2.*e_per*np.sqrt(ex**2 + ey**2 + ez**2))

    #Delta_i = (np.sqrt(1 - e_per**(-2))*(-1 + e_per**2)*(5*ex*ey - jx*jy)*eps_SA)/(e_per*np.sqrt(jx**2 + jy**2))
    Delta_i = (np.sqrt(1 - e_per**(-2))*(5*ex*(2*(-1 + e_per**2)*ey*(jx**2 + jy**2) + (-1 + 4*e_per**2)*ez*jy*jz) - jx*(2*(-1 + e_per**2)*jx**2*jy + 2*(-1 + e_per**2)*jy**3 + 5*(1 + 2*e_per**2)*ey*ez*jz + 2*(-1 + e_per**2)*jy*jz**2))*eps_SA - 15*e_per*ez*(ey*jx - ex*jy)*jz*eps_SA*np.arccos(-(1/e_per)))/(2.*e_per*np.sqrt(jx**2 + jy**2)*(jx**2 + jy**2 + jz**2))
    
    return Delta_e,Delta_i

def compute_CDA_prediction(eps_SA,e_per,ex,ey,ez,jx,jy,jz):

    Delta_e_DA,Delta_i_CDA = compute_DA_prediction(eps_SA,e_per,ex,ey,ez,jx,jy,jz)
    Delta_e_CDA = (ex*eps_SA**2*(-(np.sqrt(1 - e_per**(-2))*ey*np.arccos(-1.0/e_per)*(((5*ex**2 - 5*ey**2 - jx**2 + jy**2)*np.arccos(-(1/e_per))**4*(50148*(-5 + 2*e_per**2)*np.pi**10 + (1192325 - 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(-1031855 + 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (1329485 - 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 + 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 - 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) - 6*(-1 + e_per**2)*(5*ex*ey - jx*jy)*np.arccos(-1.0/e_per)))/2. - (3*np.arccos(-1.0/e_per)*((4*np.sqrt(1 - e_per**(-2))*jy*(-(ex*jx) + ey*jy)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + (np.sqrt(1 - e_per**(-2))*ez*(5*ey*ez - jy*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/(-1296*np.pi**14 + 22104*np.pi**12*np.arccos(-(1/e_per))**2 - 121537*np.pi**10*np.arccos(-(1/e_per))**4 + 277563*np.pi**8*np.arccos(-(1/e_per))**6 - 277563*np.pi**6*np.arccos(-(1/e_per))**8 + 121537*np.pi**4*np.arccos(-(1/e_per))**10 - 22104*np.pi**2*np.arccos(-(1/e_per))**12 + 1296*np.arccos(-(1/e_per))**14) + 3*ez*(5*ex*ez - jx*jz)*(np.sqrt(1 - e_per**(-2))*(-1 + 4*e_per**2) + 3*e_per*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per) + 6*jy*(np.sqrt(1 - e_per**(-2))*((2 + e_per**2)*ey*jx + (2 - 5*e_per**2)*ex*jy) + 3*e_per*(ey*jx - ex*jy)*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per)))/2. - (jz*np.arccos(-1.0/e_per)*((np.sqrt(1 - e_per**(-2))*(ez*jy - 5*ey*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + 3*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(ez*(jx + 8*e_per**2*jx) + (-5 + 8*e_per**2)*ex*jz) + 3*e_per*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per))))/2. + np.sqrt(1 - e_per**(-2))*(ez*((5*np.sqrt(1 - e_per**(-2))*e_per*(5*ey*ez - jy*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/(3888*np.pi**14 - 66312*np.pi**12*np.arccos(-(1/e_per))**2 + 364611*np.pi**10*np.arccos(-(1/e_per))**4 - 832689*np.pi**8*np.arccos(-(1/e_per))**6 + 832689*np.pi**6*np.arccos(-(1/e_per))**8 - 364611*np.pi**4*np.arccos(-(1/e_per))**10 + 66312*np.pi**2*np.arccos(-(1/e_per))**12 - 3888*np.arccos(-(1/e_per))**14) - (np.sqrt(1 - e_per**(-2))*(5*ey*ez - jy*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/(6.*e_per*(1296*np.pi**14 - 22104*np.pi**12*np.arccos(-(1/e_per))**2 + 121537*np.pi**10*np.arccos(-(1/e_per))**4 - 277563*np.pi**8*np.arccos(-(1/e_per))**6 + 277563*np.pi**6*np.arccos(-(1/e_per))**8 - 121537*np.pi**4*np.arccos(-(1/e_per))**10 + 22104*np.pi**2*np.arccos(-(1/e_per))**12 - 1296*np.arccos(-(1/e_per))**14)) + ((5*ex*ez - jx*jz)*(np.sqrt(1 - e_per**(-2))*(-1 + 4*e_per**2) + 3*e_per*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per))/(2.*e_per) - 5*e_per*(5*ex*ez - jx*jz)*(np.sqrt(1 - e_per**(-2))*(-1 + 4*e_per**2) + 3*e_per*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per)) + (jy - 10*e_per**2*jy)*((2*np.sqrt(1 - e_per**(-2))*(ex*jx - ey*jy)*np.arccos(-(1/e_per))**4*(50148*(-5 + 2*e_per**2)*np.pi**10 + (1192325 - 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(-1031855 + 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (1329485 - 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 + 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 - 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/(3.*e_per*(np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + ((np.sqrt(1 - e_per**(-2))*((2 + e_per**2)*ey*jx + (2 - 5*e_per**2)*ex*jy) + 3*e_per*(ey*jx - ex*jy)*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per))/e_per) + (-5 + 2*e_per**2)*((np.sqrt(1 - e_per**(-2))*ey*(((5*ex**2 - 5*ey**2 - jx**2 + jy**2)*np.arccos(-(1/e_per))**4*(50148*(-5 + 2*e_per**2)*np.pi**10 + (1192325 - 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(-1031855 + 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (1329485 - 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 + 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 - 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) - 6*(-1 + e_per**2)*(5*ex*ey - jx*jy)*np.arccos(-1.0/e_per)))/(6.*e_per) + (jz*((np.sqrt(1 - e_per**(-2))*(ez*jy - 5*ey*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + 3*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(ez*(jx + 8*e_per**2*jx) + (-5 + 8*e_per**2)*ex*jz) + 3*e_per*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per))))/(6.*e_per)))))/(4.*e_per*np.sqrt(ex**2 + ey**2 + ez**2)*np.arccos(-(1/e_per))) + (ez*eps_SA**2*(((np.sqrt(1 - e_per**(-2))*jy*(ez*jx - 5*ex*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + (np.sqrt(1 - e_per**(-2))*jx*(ez*jy - 5*ey*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + (np.sqrt(1 - e_per**(-2))*ey*(5*ex*ez - jx*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/(1296*np.pi**14 - 22104*np.pi**12*np.arccos(-(1/e_per))**2 + 121537*np.pi**10*np.arccos(-(1/e_per))**4 - 277563*np.pi**8*np.arccos(-(1/e_per))**6 + 277563*np.pi**6*np.arccos(-(1/e_per))**8 - 121537*np.pi**4*np.arccos(-(1/e_per))**10 + 22104*np.pi**2*np.arccos(-(1/e_per))**12 - 1296*np.arccos(-(1/e_per))**14) + (np.sqrt(1 - e_per**(-2))*ex*(5*ey*ez - jy*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/(1296*np.pi**14 - 22104*np.pi**12*np.arccos(-(1/e_per))**2 + 121537*np.pi**10*np.arccos(-(1/e_per))**4 - 277563*np.pi**8*np.arccos(-(1/e_per))**6 + 277563*np.pi**6*np.arccos(-(1/e_per))**8 - 121537*np.pi**4*np.arccos(-(1/e_per))**10 + 22104*np.pi**2*np.arccos(-(1/e_per))**12 - 1296*np.arccos(-(1/e_per))**14) - 3*ey*(5*ey*ez - jy*jz)*(np.sqrt(1 - e_per**(-2))*(1 + 2*e_per**2) + 3*e_per*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per) - 3*ex*(5*ex*ez - jx*jz)*(np.sqrt(1 - e_per**(-2))*(-1 + 4*e_per**2) + 3*e_per*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per) + 3*jx*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(ez*(jx + 8*e_per**2*jx) + (-5 + 8*e_per**2)*ex*jz) + 3*e_per*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per)) - 3*jy*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(ez*(jy - 10*e_per**2*jy) + (-5 + 2*e_per**2)*ey*jz) - 3*e_per*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per)))/(4.*e_per) + (np.sqrt(1 - e_per**(-2))*((2 + e_per**2)*((ey*((np.sqrt(1 - e_per**(-2))*(5*ex*ez - jx*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/(1296*np.pi**14 - 22104*np.pi**12*np.arccos(-(1/e_per))**2 + 121537*np.pi**10*np.arccos(-(1/e_per))**4 - 277563*np.pi**8*np.arccos(-(1/e_per))**6 + 277563*np.pi**6*np.arccos(-(1/e_per))**8 - 121537*np.pi**4*np.arccos(-(1/e_per))**10 + 22104*np.pi**2*np.arccos(-(1/e_per))**12 - 1296*np.arccos(-(1/e_per))**14) - 3*(5*ey*ez - jy*jz)*(np.sqrt(1 - e_per**(-2))*(1 + 2*e_per**2) + 3*e_per*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per)))/(6.*e_per) + (jx*((np.sqrt(1 - e_per**(-2))*(ez*jy - 5*ey*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + 3*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(ez*(jx + 8*e_per**2*jx) + (-5 + 8*e_per**2)*ex*jz) + 3*e_per*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per))))/(6.*e_per)) + (2 - 5*e_per**2)*(ex*(-(np.sqrt(1 - e_per**(-2))*(5*ey*ez - jy*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/(6.*e_per*(1296*np.pi**14 - 22104*np.pi**12*np.arccos(-(1/e_per))**2 + 121537*np.pi**10*np.arccos(-(1/e_per))**4 - 277563*np.pi**8*np.arccos(-(1/e_per))**6 + 277563*np.pi**6*np.arccos(-(1/e_per))**8 - 121537*np.pi**4*np.arccos(-(1/e_per))**10 + 22104*np.pi**2*np.arccos(-(1/e_per))**12 - 1296*np.arccos(-(1/e_per))**14)) + ((5*ex*ez - jx*jz)*(np.sqrt(1 - e_per**(-2))*(-1 + 4*e_per**2) + 3*e_per*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per))/(2.*e_per)) + (jy*((np.sqrt(1 - e_per**(-2))*(ez*jx - 5*ex*jz)*np.arccos(-(1/e_per))**4*(50148*(-5 + 2*e_per**2)*np.pi**10 + (1192325 - 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(-1031855 + 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (1329485 - 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 + 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 - 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + 3*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(ez*(jy - 10*e_per**2*jy) + (-5 + 2*e_per**2)*ey*jz) - 3*e_per*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per))))/(6.*e_per))))/(2.*e_per*np.arccos(-(1/e_per)))))/np.sqrt(ex**2 + ey**2 + ez**2) + (ey*eps_SA**2*((np.arccos(-1.0/e_per)*((12*np.sqrt(1 - e_per**(-2))*jx*(ex*jx - ey*jy)*np.arccos(-(1/e_per))**4*(50148*(-5 + 2*e_per**2)*np.pi**10 + (1192325 - 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(-1031855 + 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (1329485 - 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 + 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 - 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + 18*jx*(np.sqrt(1 - e_per**(-2))*((2 + e_per**2)*ey*jx + (2 - 5*e_per**2)*ex*jy) + 3*e_per*(ey*jx - ex*jy)*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per) + np.sqrt(1 - e_per**(-2))*ex*(((5*ex**2 - 5*ey**2 - jx**2 + jy**2)*np.arccos(-(1/e_per))**4*(50148*(-5 + 2*e_per**2)*np.pi**10 + (1192325 - 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(-1031855 + 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (1329485 - 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 + 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 - 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) - 6*(-1 + e_per**2)*(5*ex*ey - jx*jy)*np.arccos(-1.0/e_per)) + 3*ez*((np.sqrt(1 - e_per**(-2))*(5*ex*ez - jx*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/(1296*np.pi**14 - 22104*np.pi**12*np.arccos(-(1/e_per))**2 + 121537*np.pi**10*np.arccos(-(1/e_per))**4 - 277563*np.pi**8*np.arccos(-(1/e_per))**6 + 277563*np.pi**6*np.arccos(-(1/e_per))**8 - 121537*np.pi**4*np.arccos(-(1/e_per))**10 + 22104*np.pi**2*np.arccos(-(1/e_per))**12 - 1296*np.arccos(-(1/e_per))**14) - 3*(5*ey*ez - jy*jz)*(np.sqrt(1 - e_per**(-2))*(1 + 2*e_per**2) + 3*e_per*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per)) + jz*((np.sqrt(1 - e_per**(-2))*(ez*jx - 5*ex*jz)*np.arccos(-(1/e_per))**4*(50148*(-5 + 2*e_per**2)*np.pi**10 + (1192325 - 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(-1031855 + 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (1329485 - 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 + 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 - 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + 3*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(ez*(jy - 10*e_per**2*jy) + (-5 + 2*e_per**2)*ey*jz) - 3*e_per*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per)))))/2. + np.sqrt(1 - e_per**(-2))*(ez*((4*np.sqrt(1 - e_per**(-2))*e_per*(5*ex*ez - jx*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/(3888*np.pi**14 - 66312*np.pi**12*np.arccos(-(1/e_per))**2 + 364611*np.pi**10*np.arccos(-(1/e_per))**4 - 832689*np.pi**8*np.arccos(-(1/e_per))**6 + 832689*np.pi**6*np.arccos(-(1/e_per))**8 - 364611*np.pi**4*np.arccos(-(1/e_per))**10 + 66312*np.pi**2*np.arccos(-(1/e_per))**12 - 3888*np.arccos(-(1/e_per))**14) + (np.sqrt(1 - e_per**(-2))*(5*ex*ez - jx*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/(6.*e_per*(1296*np.pi**14 - 22104*np.pi**12*np.arccos(-(1/e_per))**2 + 121537*np.pi**10*np.arccos(-(1/e_per))**4 - 277563*np.pi**8*np.arccos(-(1/e_per))**6 + 277563*np.pi**6*np.arccos(-(1/e_per))**8 - 121537*np.pi**4*np.arccos(-(1/e_per))**10 + 22104*np.pi**2*np.arccos(-(1/e_per))**12 - 1296*np.arccos(-(1/e_per))**14)) - ((5*ey*ez - jy*jz)*(np.sqrt(1 - e_per**(-2))*(1 + 2*e_per**2) + 3*e_per*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per))/(2.*e_per) - 4*e_per*(5*ey*ez - jy*jz)*(np.sqrt(1 - e_per**(-2))*(1 + 2*e_per**2) + 3*e_per*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per)) + (jx + 8*e_per**2*jx)*((2*np.sqrt(1 - e_per**(-2))*(ex*jx - ey*jy)*np.arccos(-(1/e_per))**4*(50148*(-5 + 2*e_per**2)*np.pi**10 + (1192325 - 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(-1031855 + 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (1329485 - 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 + 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 - 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/(3.*e_per*(np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + ((np.sqrt(1 - e_per**(-2))*((2 + e_per**2)*ey*jx + (2 - 5*e_per**2)*ex*jy) + 3*e_per*(ey*jx - ex*jy)*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per))/e_per) + (-5 + 8*e_per**2)*((np.sqrt(1 - e_per**(-2))*ex*(((5*ex**2 - 5*ey**2 - jx**2 + jy**2)*np.arccos(-(1/e_per))**4*(50148*(-5 + 2*e_per**2)*np.pi**10 + (1192325 - 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(-1031855 + 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (1329485 - 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 + 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 - 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) - 6*(-1 + e_per**2)*(5*ex*ey - jx*jy)*np.arccos(-1.0/e_per)))/(6.*e_per) + (jz*((np.sqrt(1 - e_per**(-2))*(ez*jx - 5*ex*jz)*np.arccos(-(1/e_per))**4*(50148*(-5 + 2*e_per**2)*np.pi**10 + (1192325 - 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(-1031855 + 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (1329485 - 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 + 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 - 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + 3*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(ez*(jy - 10*e_per**2*jy) + (-5 + 2*e_per**2)*ey*jz) - 3*e_per*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per))))/(6.*e_per)))))/(4.*e_per*np.sqrt(ex**2 + ey**2 + ez**2)*np.arccos(-(1/e_per)))
    
    Delta_i = -((np.sqrt(jx**2 + jy**2 + jz**2)*((-((np.sqrt(1 - e_per**(-2))*(-1 + e_per**2)*(5*ex*ey - jx*jy)*eps_SA)/e_per) - (np.sqrt(1 - e_per**(-2))*(-1 + e_per**2)*eps_SA**2*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(10*(1 + 2*e_per**2)*ex*ez*jx + 10*(1 - 4*e_per**2)*ey*ez*jy + 5*(-5 + 8*e_per**2)*ex**2*jz + 5*(-5 + 2*e_per**2)*ey**2*jz + (-1 + 4*e_per**2)*jx**2*jz - (1 + 2*e_per**2)*jy**2*jz) - 3*e_per*(5*ex*ez*jx - 5*ey*ez*jy + (-jx**2 + jy**2)*jz)*np.arccos(-(1/e_per)) + 15*e_per*(3*ex*ez*jx + ex**2*jz - ey*(3*ez*jy + ey*jz))*np.arccos(-1.0/e_per)))/(4.*e_per**2*np.arccos(-(1/e_per))))/np.sqrt(jx**2 + jy**2 + jz**2) - (jz*(jz*(-((np.sqrt(1 - e_per**(-2))*(-1 + e_per**2)*(5*ex*ey - jx*jy)*eps_SA)/e_per) - (np.sqrt(1 - e_per**(-2))*(-1 + e_per**2)*eps_SA**2*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(10*(1 + 2*e_per**2)*ex*ez*jx + 10*(1 - 4*e_per**2)*ey*ez*jy + 5*(-5 + 8*e_per**2)*ex**2*jz + 5*(-5 + 2*e_per**2)*ey**2*jz + (-1 + 4*e_per**2)*jx**2*jz - (1 + 2*e_per**2)*jy**2*jz) - 3*e_per*(5*ex*ez*jx - 5*ey*ez*jy + (-jx**2 + jy**2)*jz)*np.arccos(-(1/e_per)) + 15*e_per*(3*ex*ez*jx + ex**2*jz - ey*(3*ez*jy + ey*jz))*np.arccos(-1.0/e_per)))/(4.*e_per**2*np.arccos(-(1/e_per)))) + jx*(-((5*ey*ez - jy*jz)*eps_SA*(np.sqrt(1 - e_per**(-2))*(1 + 2*e_per**2) + 3*e_per*np.arccos(-(1/e_per))))/(2.*e_per) - (eps_SA**2*(np.sqrt(1 - e_per**(-2))*(1 + 2*e_per**2) + 3*e_per*np.arccos(-(1/e_per)))*((np.sqrt(1 - e_per**(-2))*jy*(((5*ex**2 - 5*ey**2 - jx**2 + jy**2)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + 6*(-1 + e_per**2)*(5*ex*ey - jx*jy)*np.arccos(-1.0/e_per)))/(6.*e_per) + (jz*((np.sqrt(1 - e_per**(-2))*(5*ey*ez - jy*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/(1296*np.pi**14 - 22104*np.pi**12*np.arccos(-(1/e_per))**2 + 121537*np.pi**10*np.arccos(-(1/e_per))**4 - 277563*np.pi**8*np.arccos(-(1/e_per))**6 + 277563*np.pi**6*np.arccos(-(1/e_per))**8 - 121537*np.pi**4*np.arccos(-(1/e_per))**10 + 22104*np.pi**2*np.arccos(-(1/e_per))**12 - 1296*np.arccos(-(1/e_per))**14) - 3*(5*ex*ez - jx*jz)*(np.sqrt(1 - e_per**(-2))*(-1 + 4*e_per**2) + 3*e_per*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per)))/(6.*e_per) + 5*ey*((2*np.sqrt(1 - e_per**(-2))*(ex*jx - ey*jy)*np.arccos(-(1/e_per))**4*(50148*(-5 + 2*e_per**2)*np.pi**10 + (1192325 - 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(-1031855 + 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (1329485 - 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 + 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 - 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/(3.*e_per*(np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + ((np.sqrt(1 - e_per**(-2))*((2 + e_per**2)*ey*jx + (2 - 5*e_per**2)*ex*jy) + 3*e_per*(ey*jx - ex*jy)*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per))/e_per) + (5*ez*((np.sqrt(1 - e_per**(-2))*(ez*jy - 5*ey*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + 3*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(ez*(jx + 8*e_per**2*jx) + (-5 + 8*e_per**2)*ex*jz) + 3*e_per*(3*ez*jx + ex*jz)*np.arccos(-1.0/e_per))))/(6.*e_per)))/(4.*e_per*np.arccos(-(1/e_per)))) + jy*(((5*ex*ez - jx*jz)*eps_SA*(np.sqrt(1 - e_per**(-2))*(-1 + 4*e_per**2) + 3*e_per*np.arccos(-(1/e_per))))/(2.*e_per) + (eps_SA**2*(np.sqrt(1 - e_per**(-2))*(-1 + 4*e_per**2) + 3*e_per*np.arccos(-(1/e_per)))*((np.sqrt(1 - e_per**(-2))*jx*(((5*ex**2 - 5*ey**2 - jx**2 + jy**2)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + 6*(-1 + e_per**2)*(5*ex*ey - jx*jy)*np.arccos(-1.0/e_per)))/(6.*e_per) + (jz*((np.sqrt(1 - e_per**(-2))*(5*ex*ez - jx*jz)*np.arccos(-(1/e_per))**4*(-50148*(-5 + 2*e_per**2)*np.pi**10 + (-1192325 + 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(1031855 - 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (-1329485 + 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 - 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 + 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/(-1296*np.pi**14 + 22104*np.pi**12*np.arccos(-(1/e_per))**2 - 121537*np.pi**10*np.arccos(-(1/e_per))**4 + 277563*np.pi**8*np.arccos(-(1/e_per))**6 - 277563*np.pi**6*np.arccos(-(1/e_per))**8 + 121537*np.pi**4*np.arccos(-(1/e_per))**10 - 22104*np.pi**2*np.arccos(-(1/e_per))**12 + 1296*np.arccos(-(1/e_per))**14) + 3*(5*ey*ez - jy*jz)*(np.sqrt(1 - e_per**(-2))*(1 + 2*e_per**2) + 3*e_per*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per)))/(6.*e_per) + 5*ex*((2*np.sqrt(1 - e_per**(-2))*(ex*jx - ey*jy)*np.arccos(-(1/e_per))**4*(50148*(-5 + 2*e_per**2)*np.pi**10 + (1192325 - 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(-1031855 + 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (1329485 - 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 + 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 - 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/(3.*e_per*(np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + ((np.sqrt(1 - e_per**(-2))*((2 + e_per**2)*ey*jx + (2 - 5*e_per**2)*ex*jy) + 3*e_per*(ey*jx - ex*jy)*np.arccos(-(1/e_per)))*np.arccos(-1.0/e_per))/e_per) + (5*ez*((np.sqrt(1 - e_per**(-2))*(ez*jx - 5*ex*jz)*np.arccos(-(1/e_per))**4*(50148*(-5 + 2*e_per**2)*np.pi**10 + (1192325 - 761624*e_per**2)*np.pi**8*np.arccos(-(1/e_per))**2 + 2*(-1031855 + 873044*e_per**2)*np.pi**6*np.arccos(-(1/e_per))**4 + (1329485 - 1307672*e_per**2)*np.pi**4*np.arccos(-(1/e_per))**6 + 144*(-2075 + 2564*e_per**2)*np.pi**2*np.arccos(-(1/e_per))**8 - 3888*(-5 + 8*e_per**2)*np.arccos(-(1/e_per))**10))/((np.pi**2 - np.arccos(-(1/e_per))**2)*(81*np.pi**4 - 45*np.pi**2*np.arccos(-(1/e_per))**2 + 4*np.arccos(-(1/e_per))**4)*(16*np.pi**4 - 40*np.pi**2*np.arccos(-(1/e_per))**2 + 9*np.arccos(-(1/e_per))**4)*(np.pi**4 - 13*np.pi**2*np.arccos(-(1/e_per))**2 + 36*np.arccos(-(1/e_per))**4)) + 3*np.arccos(-1.0/e_per)*(np.sqrt(1 - e_per**(-2))*(ez*(jy - 10*e_per**2*jy) + (-5 + 2*e_per**2)*ey*jz) - 3*e_per*(3*ez*jy + ey*jz)*np.arccos(-1.0/e_per))))/(6.*e_per)))/(4.*e_per*np.arccos(-(1/e_per))))))/(jx**2 + jy**2 + jz**2)**1.5))/np.sqrt(jx**2 + jy**2))
    
    return Delta_e_DA+Delta_e_CDA,Delta_i
    
def plot_function_series(args,data_series):
    Q_points,Delta_es,Delta_is = data_series
    
    M = args.M
    M_per = args.M_per
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
        eps_SA = compute_eps_SA(M,M_per,a_div_Q,e_per)
        
        Delta_e_DA,Delta_i_DA = compute_DA_prediction(eps_SA,e_per,ex,ey,ez,jx,jy,jz)
        Delta_e_CDA,Delta_i_CDA = compute_CDA_prediction(eps_SA,e_per,ex,ey,ez,jx,jy,jz)
        Delta_es_DA.append(Delta_e_DA)
        Delta_es_CDA.append(Delta_e_CDA)

        Delta_is_DA.append(Delta_i_DA)
        Delta_is_CDA.append(Delta_i_CDA)
       # print 'Delta_i_DA',Delta_i_DA
        
    Delta_es_DA = np.array(Delta_es_DA)
    Delta_es_CDA = np.array(Delta_es_CDA)
    Delta_is_DA = np.array(Delta_is_DA)
    Delta_is_CDA = np.array(Delta_is_CDA)

    fontsize=18
    labelsize=12
    
    fig=pyplot.figure(figsize=(8,10))
    plot1=fig.add_subplot(2,1,1,yscale="log",xscale="log")
    plot2=fig.add_subplot(2,1,2,yscale="linear",xscale="log")
    #plot3=fig.add_subplot(3,1,3)
    
    indices_pos = [i for i in range(len(Delta_es)) if Delta_es[i] >= 0.0]
    indices_neg = [i for i in range(len(Delta_es)) if Delta_es[i] < 0.0]
    
    s=10
    plot1.scatter(Q_div_a_points[indices_pos],Delta_es[indices_pos],color='b',s=s, facecolors='none',label="$\mathrm{SA}$")
    plot1.scatter(Q_div_a_points[indices_neg],-Delta_es[indices_neg],color='r',s=s, facecolors='none')


    indices_pos = [i for i in range(len(Delta_is)) if Delta_is[i] >= 0.0]
    indices_neg = [i for i in range(len(Delta_is)) if Delta_is[i] < 0.0]
    
    plot2.scatter(Q_div_a_points[indices_pos],Delta_is[indices_pos]*180.0/np.pi,color='b',s=s, facecolors='none')
    plot2.scatter(Q_div_a_points[indices_neg],-Delta_is[indices_neg]*180.0/np.pi,color='r',s=s, facecolors='none')

    w=2.0
    plot1.axhline(y = 1.0 - e,color='k',linestyle='dotted',linewidth=w,label="$1-e$")
    plot1.plot(plot_Q_div_a_points,Delta_es_DA,color='k',linestyle='dashed',linewidth=w,label="$\mathrm{DA}$")

    indices_pos = [i for i in range(len(Delta_es_CDA)) if Delta_es_CDA[i] >= 0.0]
    indices_neg = [i for i in range(len(Delta_es_CDA)) if Delta_es_CDA[i] < 0.0]
    
    plot1.plot(plot_Q_div_a_points[indices_pos],Delta_es_CDA[indices_pos],color='b',linestyle='solid',linewidth=w,label="$\mathrm{CDA}$")
    plot1.plot(plot_Q_div_a_points[indices_neg],-Delta_es_CDA[indices_neg],color='r',linestyle='solid',linewidth=w)    


    indices_pos = [i for i in range(len(Delta_is_DA)) if Delta_is_DA[i] >= 0.0]
    indices_neg = [i for i in range(len(Delta_is_DA)) if Delta_is_DA[i] < 0.0]

    #plot2.plot(plot_Q_div_a_points[indices_pos],Delta_is_DA[indices_pos]*180.0/np.pi,color='b',linestyle='dashed',linewidth=w)
    #plot2.plot(plot_Q_div_a_points[indices_neg],-Delta_is_DA[indices_neg]*180.0/np.pi,color='r',linestyle='dashed',linewidth=w)



    indices_pos = [i for i in range(len(Delta_is_CDA)) if Delta_is_CDA[i] >= 0.0]
    indices_neg = [i for i in range(len(Delta_is_CDA)) if Delta_is_CDA[i] < 0.0]

    #plot2.plot(plot_Q_div_a_points[indices_pos],Delta_is_CDA[indices_pos]*180.0/np.pi,color='b',linestyle='solid',linewidth=w)
    #plot2.plot(plot_Q_div_a_points[indices_neg],-Delta_is_CDA[indices_neg]*180.0/np.pi,color='r',linestyle='solid',linewidth=w)
    
    plots = [plot1,plot2]
    labels = [r"$\Delta e$",r"$\Delta i/\mathrm{deg}$"]
    for index,plot in enumerate(plots):
        if index==1:
            plot.set_xlabel(r"$Q/a$",fontsize=fontsize)
        plot.set_ylabel(labels[index],fontsize=fontsize)
                
        plot.tick_params(axis='both', which ='major', labelsize = labelsize)

        if index in [0]:
            plot.set_xticklabels([])
    #plot3.axhline(y=1.0,linestyle='dotted',color='k')

    plot1.set_title("$e=%s;\,i=%s\,\mathrm{deg};\, \omega=%s\,\mathrm{deg};\, \Omega=%s\,\mathrm{deg}$"%(args.e,round(args.i*180.0/np.pi,1),round(args.omega*180.0/np.pi,1),round(args.Omega*180.0/np.pi,1)),fontsize=0.8*fontsize)

    handles,labels = plot1.get_legend_handles_labels()
    plot1.legend(handles,labels,loc="upper right",fontsize=0.8*fontsize)
    
    fig.subplots_adjust(hspace=0.0,wspace=0.0)
    filename = 'Deltas_' + str(args.name) + '.pdf'
    fig.savefig(filename,dpi=200)
    
    pyplot.show()
        
if __name__ == '__main__':
    args = parse_arguments()

    if args.calc==1:
        data = integrate(args)

        if args.plot == True:
            if HAS_MATPLOTLIB == False:
                print 'Error importing Matplotlib -- not making plot'
                exit(-1)
            plot_function(args,data)

    if args.calc==2:
        data_series = integrate_series(args)

        if args.plot == True:
            if HAS_MATPLOTLIB == False:
                print 'Error importing Matplotlib -- not making plot'
                exit(-1)
            plot_function_series(args,data_series)

