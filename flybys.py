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

#from wrapperflybyslibrary import flybyslibrary
import core

try:
    from matplotlib import pyplot
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def mkdir_p(path):
    import os,errno
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
        
def add_bool_arg(parser, name, default=False,help=None):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true',help="Enable %s"%help)
    group.add_argument('--no-' + name, dest=name, action='store_false',help="Disable %s"%help)
    parser.set_defaults(**{name:default})

def parse_arguments():
    

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--mode",                           type=float,     dest="mode",                        default=1,              help="mode -- 1: single integration; 2: single integration (illustrating Fourier series); 3: series integration; 4: series integration (with detailed time plots); 5: series integration (with different orbital angles); 6: series integration (in the context of 1PN terms); 7: make overview plot of importance of SO terms.")
    parser.add_argument("--name",                           type=str,       dest="name",                        default="test01",       help="name")
    parser.add_argument("--m1",                             type=float,     dest="m1",                          default=1.0,            help="Primary mass")
    parser.add_argument("--m2",                             type=float,     dest="m2",                          default=1.0,            help="Secondary mass")
    parser.add_argument("--M_per",                          type=float,     dest="M_per",                       default=1.0,            help="Perturber mass")
    parser.add_argument("--e_per",                          type=float,     dest="e_per",                       default=1.0+1.0e-15,    help="Perturber eccentricity")
    parser.add_argument("--Q",                              type=float,     dest="Q",                           default=4.0,            help="Perturber periapsis distance (same units as a)")
    parser.add_argument("--Q_min",                          type=float,     dest="Q_min",                       default=2.0,            help="Minimum perturber periapsis distance (in case of series)")
    parser.add_argument("--Q_max",                          type=float,     dest="Q_max",                       default=50.0,           help="Maximum perturber periapsis distance (in case of series)")
    parser.add_argument("--N_Q",                            type=int,       dest="N_Q",                         default=200,            help="Number of systems in series")
    parser.add_argument("--N_AP",                           type=int,       dest="N_AP",                        default=10,             help="Number of systems of argument of periapsis for each point Q in PN series")
    parser.add_argument("--a",                              type=float,     dest="a",                           default=1.0,            help="Binary semimajor axis (same units as Q)")    
    parser.add_argument("--e",                              type=float,     dest="e",                           default=0.999,          help="Binary eccentricity")    
    parser.add_argument("--i",                              type=float,     dest="i",                           default=np.pi/2.0,      help="Binary inclination")    
    parser.add_argument("--omega",                          type=float,     dest="omega",                       default=np.pi/4.0,      help="Binary argument of periapsis")
    parser.add_argument("--Omega",                          type=float,     dest="Omega",                       default=-np.pi/4.0,     help="Binary longitude of the ascending node")
    parser.add_argument("--N_steps",                        type=int,       dest="N_steps",                     default=100,            help="Number of external output steps taken by odeint")
    parser.add_argument("--mxstep",                         type=int,       dest="mxstep",                      default=1000000,        help="Maximum number of internal steps taken in the ODE integration. Increase if ODE integrator give mstep errors. ")    
    parser.add_argument("--theta_bin",                      type=float,     dest="theta_bin",                   default=1.0,            help="Initial binary true anomaly (3-body integration only)")
    parser.add_argument("--fraction_theta_0",               type=float,     dest="fraction_theta_0",            default=0.95,           help="Initial perturber true anomaly (3-body only), expressed as a fraction of -\arccos(-1/e_per). Default=0.9; increase if 3-body integrations do not seem converged. ")
    parser.add_argument("--G",                              type=float,     dest="G",                           default=4.0*np.pi**2,   help="Gravitational constant used in 3-body integrations. Should not affect Newtonian results. ")
    parser.add_argument("--c",                              type=float,     dest="c",                           default=63239.72638679138, help="Speed of light (PN terms only). ")
    parser.add_argument("--fontsize",                       type=float,     dest="fontsize",                    default=22,             help="Fontsize for plots")
    parser.add_argument("--labelsize",                      type=float,     dest="labelsize",                   default=16,             help="Labelsize for plots")
    parser.add_argument("--ymin",                           type=float,     dest="ymin",                        default=1.0e-5,         help="ymin for series plots")
    parser.add_argument("--ymax",                           type=float,     dest="ymax",                        default=1.0e0,          help="ymax for series plots")
    parser.add_argument("--xmin",                           type=float,     dest="xmin",                        default=2.0e0,          help="xmin for series plots")
    parser.add_argument("--xmax",                           type=float,     dest="xmax",                        default=5.0e1,          help="xmax for series plots")

    ### boolean arguments ###
    add_bool_arg(parser, 'verbose',                         default=False,          help="Verbose terminal output")
    add_bool_arg(parser, 'calc',                            default=True,           help="Do calculation (and save results). If False, will try to load previous results")
    add_bool_arg(parser, 'plot',                            default=True,           help="Make plots")
    add_bool_arg(parser, 'plot_fancy',                      default=False,          help="Use LaTeX for plot labels (slower)")
    add_bool_arg(parser, 'show',                            default=True,           help="Show plots")
    add_bool_arg(parser, 'include_quadrupole_terms',        default=True,           help="Include quadrupole-order terms")
    add_bool_arg(parser, 'include_octupole_terms',          default=True,           help="include octupole-order terms")
    add_bool_arg(parser, 'include_1PN_terms',               default=False,          help="include 1PN terms")
    add_bool_arg(parser, 'do_nbody',                        default=False,          help="Do 3-body integrations as well as SA")
    add_bool_arg(parser, 'include_analytic_FO_terms',       default=True,           help="include analytic first-order terms in epsilon_SA")
    add_bool_arg(parser, 'include_analytic_SO_terms',       default=True,           help="include analytic second-order terms in epsilon_SA")
    add_bool_arg(parser, 'include_analytic_TO_terms',       default=False,          help="include analytic third-order terms in epsilon_SA")
    #add_bool_arg(parser, 'use_c',                           default=False,          help="Use c for analytic functions (implementation needs to be updated)")
    add_bool_arg(parser, 'include_quadrupole_only_lines',   default=False,          help="Show quadrupole-order only analytic lines (mode 3)")
    add_bool_arg(parser, 'show_inset',                      default=False,          help="Show inset plot (mode 3)")
    
    args = parser.parse_args()
               
    args.m = args.m1 + args.m2
    args.use_c = False ### needs to be updated; disabled for the moment
        
    args.data_filename = os.path.dirname(os.path.realpath(__file__)) + '/data/data_' + str(args.name) + '_m1_' + str(args.m1) + '_m2_' + str(args.m2) + '_M_per_' + str(args.M_per) + '_e_per_' \
        + str(args.e_per) + '_Q_' + str(args.Q) + '_a_' + str(args.a) + '_e_' + str(args.e) + '_i_' + str(args.i) + '_omega_' + str(args.omega) + '_Omega_' + str(args.Omega) \
        + '_do_nbody_' + str(args.do_nbody) + '.pkl'
    args.series_data_filename = os.path.dirname(os.path.realpath(__file__)) + '/data/series_data_' + str(args.name) + '_m1_' + str(args.m1) + '_m2_' + str(args.m2) + '_M_per_' \
        + str(args.M_per) + '_e_per_' + str(args.e_per) + '_Q_' + str(args.Q) + '_a_' + str(args.a) + '_e_' + str(args.e) + '_i_' + str(args.i) + '_omega_' + str(args.omega) + '_Omega_' \
        + str(args.Omega) + '_Q_min_' + str(args.Q_min) + '_Q_max_' + str(args.Q_max) + '_N_Q_' + str(args.N_Q) \
        + '_do_nbody_' + str(args.do_nbody) + '.pkl'
    args.series_angles_data_filename = os.path.dirname(os.path.realpath(__file__)) + '/data/series_angles_data_' + str(args.name) + '_m1_' + str(args.m1) + '_m2_' + str(args.m2) + '_M_per_' \
        + str(args.M_per) + '_e_per_' + str(args.e_per) + '_Q_' + str(args.Q) + '_a_' + str(args.a) + '_e_' + str(args.e) + '_i_' + str(args.i) + '_omega_' + str(args.omega) + '_Omega_' \
        + str(args.Omega) + '_Q_min_' + str(args.Q_min) + '_Q_max_' + str(args.Q_max) + '_N_Q_' + str(args.N_Q) \
        + '_do_nbody_' + str(args.do_nbody) + '.pkl'
    args.series_PN_data_filename = os.path.dirname(os.path.realpath(__file__)) + '/data/series_PN_data_' + str(args.name) + '_m1_' + str(args.m1) + '_m2_' + str(args.m2) + '_M_per_' \
        + str(args.M_per) + '_e_per_' + str(args.e_per) + '_Q_' + str(args.Q) + '_a_' + str(args.a) + '_e_' + str(args.e) + '_i_' + str(args.i) + '_omega_' + str(args.omega) + '_Omega_' \
        + str(args.Omega) + '_Q_min_' + str(args.Q_min) + '_Q_max_' + str(args.Q_max) + '_N_Q_' + str(args.N_Q) \
        + '_do_nbody_' + str(args.do_nbody) + '_include_1PN_terms_' + str(args.include_1PN_terms) + '_fraction_theta_0_' + str(args.fraction_theta_0) + '_N_AP_' + str(args.N_AP) + '.pkl'

    mkdir_p(os.path.dirname(os.path.realpath(__file__)) + '/data')
    mkdir_p(os.path.dirname(os.path.realpath(__file__)) + '/figs')
    
    return args
    

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
        data = core.integrate(args)
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

def integrate_series_angles(args):
    i0 = args.i
    AP0 = args.omega
    LAN0 = args.Omega
    
    i_points = np.linspace(0.0,np.pi,args.N_Q)
    AP_points = np.linspace(0.0,2.0*np.pi,args.N_Q)
    LAN_points = np.linspace(0.0,2.0*np.pi,args.N_Q)

    Delta_es_i = []
    Delta_is_i = []

    Delta_es_AP = []
    Delta_is_AP = []

    Delta_es_LAN = []
    Delta_is_LAN = []
    
    nbody_Delta_es_i = []
    nbody_Delta_is_i = []
    nbody_Delta_es_AP = []
    nbody_Delta_is_AP = []
    nbody_Delta_es_LAN = []
    nbody_Delta_is_LAN = []

    do_nbody = False

    for index_i,i in enumerate(i_points):
        args.i = i
        args.omega = AP0
        args.Omega = LAN0
        data = core.integrate(args)
        
        if data["do_nbody"] == True:
            nbody_Delta_es_i.append(data["nbody_Delta_e"])
            nbody_Delta_is_i.append(data["nbody_Delta_i"])
            do_nbody = True
        
        Delta_es_i.append(data["Delta_e"])
        Delta_is_i.append(data["Delta_i"])
    
    Delta_es_i = np.array(Delta_es_i)
    Delta_is_i = np.array(Delta_is_i)
    nbody_Delta_es_i = np.array(nbody_Delta_es_i)
    nbody_Delta_is_i = np.array(nbody_Delta_is_i)
    
    for index_AP,AP in enumerate(AP_points):
        args.i = i0
        args.omega = AP
        args.Omega = LAN0
        data = core.integrate(args)
        
        if data["do_nbody"] == True:
            nbody_Delta_es_AP.append(data["nbody_Delta_e"])
            nbody_Delta_is_AP.append(data["nbody_Delta_i"])
            do_nbody = True
        
        Delta_es_AP.append(data["Delta_e"])
        Delta_is_AP.append(data["Delta_i"])
    
    Delta_es_AP = np.array(Delta_es_AP)
    Delta_is_AP = np.array(Delta_is_AP)
    nbody_Delta_es_AP = np.array(nbody_Delta_es_AP)
    nbody_Delta_is_AP = np.array(nbody_Delta_is_AP)
    
    for index_LAN,LAN in enumerate(LAN_points):
        args.i = i0
        args.omega = AP0
        args.Omega = LAN
        data = core.integrate(args)
        
        if data["do_nbody"] == True:
            nbody_Delta_es_LAN.append(data["nbody_Delta_e"])
            nbody_Delta_is_LAN.append(data["nbody_Delta_i"])
            do_nbody = True
        
        Delta_es_LAN.append(data["Delta_e"])
        Delta_is_LAN.append(data["Delta_i"])
    
    Delta_es_LAN = np.array(Delta_es_LAN)
    Delta_is_LAN = np.array(Delta_is_LAN)
    nbody_Delta_es_LAN = np.array(nbody_Delta_es_LAN)
    nbody_Delta_is_LAN = np.array(nbody_Delta_is_LAN)
    
    data_series_angles = {'i_points':i_points,'AP_points':AP_points,'LAN_points':LAN_points, \
        'Delta_es_i':Delta_es_i,'Delta_is_i':Delta_is_i, \
        'Delta_es_AP':Delta_es_AP,'Delta_is_AP':Delta_is_AP, \
        'Delta_es_LAN':Delta_es_LAN,'Delta_is_LAN':Delta_is_LAN, \
        'do_nbody':do_nbody,'nbody_Delta_es_i':nbody_Delta_es_i,'nbody_Delta_is_i':nbody_Delta_is_i, \
        'nbody_Delta_es_AP':nbody_Delta_es_AP,'nbody_Delta_is_AP':nbody_Delta_is_AP,
        'nbody_Delta_es_LAN':nbody_Delta_es_LAN,'nbody_Delta_is_LAN':nbody_Delta_is_LAN}

    filename = args.series_angles_data_filename

    with open(filename, 'wb') as handle:
        pickle.dump(data_series_angles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    args.i = i0
    args.omega = AP0
    args.Omega = LAN0
    
    
    return data_series_angles


def integrate_series_PN(args):
    
    Q_points = pow(10.0,np.linspace(np.log10(args.Q_min),np.log10(args.Q_max),args.N_Q))
    Delta_es = []
    Delta_is = []
    Delta_es_mean = []
    Delta_is_mean = []
    Delta_es_rms = []
    Delta_is_rms = []
    
    AP_points = np.linspace(0.0,2.0*np.pi,args.N_AP)
    
    nbody_Delta_es = []
    nbody_Delta_is = []
    do_nbody = False
    thetas_Q = []
    es_Q = []
    is_Q = []
    for index_Q,Q in enumerate(Q_points):
        args.Q = Q
        
        Delta_es_AP = []
        Delta_is_AP = []
        for index_AP,AP in enumerate(AP_points):
            args.omega = AP
            
            data = core.integrate(args)
            Delta_e = data["Delta_e"]
            Delta_i = data["Delta_i"]
            
#            if data["do_nbody"] == True:
#                nbody_Delta_es.append(data["nbody_Delta_e"])
#                nbody_Delta_is.append(data["nbody_Delta_i"])
#                do_nbody = True
        
            Delta_es_AP.append(Delta_e)
            Delta_is_AP.append(Delta_i)
            print( 'index_Q',index_Q,'Delta e',Delta_e)
        
        if 1==0:
            fig=pyplot.figure()
            plot=fig.add_subplot(1,1,1)
            plot.hist(Delta_es_AP,histtype='step')
            pyplot.show()
        
        Delta_es_AP = np.array(Delta_es_AP)
        
        Delta_es_mean.append(np.mean(np.array(Delta_es_AP)))
        Delta_is_mean.append(np.mean(np.array(Delta_is_AP)))
        Delta_es_rms.append(np.sqrt(np.mean(np.array(Delta_es_AP)**2)))
        Delta_is_rms.append(np.sqrt(np.mean(np.array(Delta_is_AP)**2)))
        
    Delta_es_mean = np.array(Delta_es_mean)
    Delta_is_mean = np.array(Delta_is_mean)
    Delta_es_rms = np.array(Delta_es_rms)
    Delta_is_rms = np.array(Delta_is_rms)
    
#    nbody_Delta_es = np.array(nbody_Delta_es)
#    nbody_Delta_is = np.array(nbody_Delta_is)
    
    data_series = {'Q_points':Q_points,'Delta_es_mean':Delta_es_mean,'Delta_is_mean':Delta_is_mean, \
        'Delta_es_rms':Delta_es_rms,'Delta_is_rms':Delta_is_rms}

    filename = args.series_PN_data_filename

    with open(filename, 'wb') as handle:
        pickle.dump(data_series, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data_series

    
def plot_function(args,data):
    a = args.a
    e = args.e
    thetas = data["thetas"]
    e_sol = data["e_sol"]
    j_sol = data["j_sol"]
    i_sol = data["i_sol"]
    Delta_e = data["Delta_e"]
    Delta_i = data["Delta_i"]
    
    print( 'Delta_e',Delta_e)
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
                
        plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)

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
    
    ex0,ey0,ez0,jx0,jy0,jz0 = core.orbital_elements_to_orbital_vectors(e,i,omega,Omega)

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
        
        print( 'Delta_e',Delta_e)
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

        plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)

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
    
    ex,ey,ez,jx,jy,jz = core.orbital_elements_to_orbital_vectors(e,i,omega,Omega)
    
    Delta_es_FO,Delta_is_FO = [],[]
    Delta_es_SO,Delta_is_SO = [],[]
    Delta_es_TO,Delta_is_TO = [],[]
    
    Delta_es_SO_no_oct,Delta_is_SO_no_oct = [],[]
    
    plot_Q_div_a_points = pow(10.0,np.linspace(np.log10(np.amin(Q_div_a_points)),np.log10(np.amax(Q_div_a_points)),2000))
    
    Delta_es_1PN = []
    
    for index,Q_div_a in enumerate(plot_Q_div_a_points):
        a_div_Q = 1.0/Q_div_a
        eps_SA = core.compute_eps_SA(m,M_per,a_div_Q,e_per)
        eps_oct = core.compute_eps_oct(m1,m2,m,a_div_Q,e_per)
        
        if args.include_analytic_FO_terms == True:
            Delta_e_FO,Delta_i_FO = core.compute_FO_prediction(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz)

            Delta_es_FO.append(Delta_e_FO)
            Delta_is_FO.append(Delta_i_FO)    

        if args.include_analytic_SO_terms == True:
            Delta_e_SO,Delta_i_SO = core.compute_SO_prediction(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz)
            
            Delta_es_SO.append(Delta_e_SO)
            Delta_is_SO.append(Delta_i_SO)
            
        if args.include_analytic_TO_terms == True:
            Delta_e_TO_only,Delta_i_TO_only = core.compute_TO_prediction_only(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz)

            Delta_e_TO = Delta_e_SO + Delta_e_TO_only
            Delta_i_TO = Delta_i_SO

            Delta_es_TO.append(Delta_e_TO)
            Delta_is_TO.append(Delta_i_TO)


        if args.include_quadrupole_only_lines == True:
            prev = args.include_octupole_terms
            args.include_octupole_terms = False
            Delta_e_SO,Delta_i_SO = core.compute_SO_prediction(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz)
            Delta_es_SO_no_oct.append(Delta_e_SO)
            Delta_is_SO_no_oct.append(Delta_i_SO)
            args.include_octupole_terms = prev

        g1 = Delta_e_FO/eps_SA
        g2 = (Delta_e_SO-Delta_e_FO)/(eps_SA**2)
    
    a_div_Q_crit = pow( (-g1/g2)*(np.sqrt(m*(m+M_per))/M_per),2.0/3.0)*(1.0+e_per)
    Q_div_a_crit = 1.0/a_div_Q_crit

    Q_div_a_crit2 = Q_div_a_crit*pow(0.5,-2.0/3.0)


    Q_div_a_crit_R_unity = pow(0.5,-2.0/3.0)*pow( (1.0 + M_per/m)*(1.0 + e_per), 1.0/3.0)


    f_TB_omega = f_TB_omega_function(i,omega,Omega)

    rg = args.G*args.m/(args.c**2)

    Delta_es_FO = np.array(Delta_es_FO)
    Delta_es_SO = np.array(Delta_es_SO)
    Delta_es_TO = np.array(Delta_es_TO)
    Delta_is_FO = np.array(Delta_is_FO)
    Delta_is_SO = np.array(Delta_is_SO)
    Delta_is_TO = np.array(Delta_is_TO)
    Delta_es_SO_no_oct = np.array(Delta_es_SO_no_oct)
    Delta_is_SO_no_oct = np.array(Delta_is_SO_no_oct)
    
    Delta_es_1PN = np.array(Delta_es_1PN)
    
    fontsize=args.fontsize
    labelsize=args.labelsize
    
    fig=pyplot.figure(figsize=(8,10))
    plot1=fig.add_subplot(2,1,1,yscale="log",xscale="log")
    plot2=fig.add_subplot(2,1,2,yscale="linear",xscale="log")
    fig_c=pyplot.figure(figsize=(8,6))
    plot_c=fig_c.add_subplot(1,1,1,yscale="log",xscale="log")
    

    if args.show_inset == True:
        from mpl_toolkits.axes_grid.inset_locator import inset_axes
        plot1_inset = inset_axes(plot_c,
        width="25%", # width = 30% of parent_bbox
        height=1.4, # height : 1 inch
        loc="lower right")
        plot1_inset.set_xscale("log")
        plot1_inset.set_yscale("log")

    s_nbody=50
    s=10
    if args.show_inset == True:
        plots = [plot1,plot_c,plot1_inset]
    else:
        plots = [plot1,plot_c]
    for plot in plots:
        #plot.axvline(x=Q_div_a_crit,color='k',linestyle='dotted')
        #plot.axvline(x=Q_div_a_crit2,color='k',linestyle='dotted')
        plot.axvline(x=Q_div_a_crit_R_unity,color='g',linestyle='dotted')
    
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
        
        if len(Delta_es_FO)>0:
            indices_pos = [i for i in range(len(Delta_es_FO)) if Delta_es_FO[i] >= 0.0]
            indices_neg = [i for i in range(len(Delta_es_FO)) if Delta_es_FO[i] < 0.0]
            plot.plot(plot_Q_div_a_points[indices_pos],Delta_es_FO[indices_pos],color='b',linestyle='dashed',linewidth=w,label="$\mathrm{FO}$")
            plot.plot(plot_Q_div_a_points[indices_neg],-Delta_es_FO[indices_neg],color='r',linestyle='dashed',linewidth=w)

        if len(Delta_es_SO)>0:
            indices_pos = [i for i in range(len(Delta_es_SO)) if Delta_es_SO[i] >= 0.0]
            indices_neg = [i for i in range(len(Delta_es_SO)) if Delta_es_SO[i] < 0.0]
            plot.plot(plot_Q_div_a_points[indices_pos],Delta_es_SO[indices_pos],color='b',linestyle='solid',linewidth=w,label="$\mathrm{SO}$")
            plot.plot(plot_Q_div_a_points[indices_neg],-Delta_es_SO[indices_neg],color='r',linestyle='solid',linewidth=w)    

        if len(Delta_es_TO)>0:
            indices_pos = [i for i in range(len(Delta_es_TO)) if Delta_es_TO[i] >= 0.0]
            indices_neg = [i for i in range(len(Delta_es_TO)) if Delta_es_TO[i] < 0.0]
            plot.plot(plot_Q_div_a_points[indices_pos],Delta_es_TO[indices_pos],color='b',linestyle='dotted',linewidth=w,label="$\mathrm{TO}$")
            plot.plot(plot_Q_div_a_points[indices_neg],-Delta_es_TO[indices_neg],color='r',linestyle='dotted',linewidth=w)    
        
        
        if args.include_quadrupole_only_lines == True:
            indices_pos = [i for i in range(len(Delta_es_SO_no_oct)) if Delta_es_SO_no_oct[i] >= 0.0]
            indices_neg = [i for i in range(len(Delta_es_SO_no_oct)) if Delta_es_SO_no_oct[i] < 0.0]
            plot.plot(plot_Q_div_a_points[indices_pos],Delta_es_SO_no_oct[indices_pos],color='k',linestyle='dotted',linewidth=w,label="$\mathrm{SO\,(no\,oct)}$")
            plot.plot(plot_Q_div_a_points[indices_neg],-Delta_es_SO_no_oct[indices_neg],color='k',linestyle='dotted',linewidth=w)    
            
        
        if len(Delta_es_1PN)>0:
            indices_pos = [i for i in range(len(Delta_es_TO)) if Delta_es_1PN[i] >= 0.0]
            indices_neg = [i for i in range(len(Delta_es_TO)) if Delta_es_1PN[i] < 0.0]
            plot.plot(plot_Q_div_a_points[indices_pos],Delta_es_1PN[indices_pos],color='b',linestyle='-.',linewidth=w,label="$\mathrm{TO}$")
            plot.plot(plot_Q_div_a_points[indices_neg],-Delta_es_1PN[indices_neg],color='r',linestyle='-.',linewidth=w)            

        if 1==0:
            alphas = [0.1,0.5,1.0,2.0,10.0]
            lw=0.5
            for alpha in alphas:
                Q_div_a_PN = np.sqrt(1.0-args.e**2)*pow( (1.0/64.0)*np.fabs(f_TB_omega)*alpha*(args.a/rg)*(args.M_per/args.m), 1.0/3.0)
                Q_div_a_PN2 = pow( ((1.0-args.e**2)/(3.0*rg/args.a))**2*( ((args.m+args.M_per)/args.m)*(1.0+args.e_per)),1.0/3.0)
                #print 'Q_div_a_PN',Q_div_a_PN,'f_TB_omega',f_TB_omega

                plot.axvline(x = Q_div_a_PN,color='r',linestyle='dotted',linewidth=lw)
                plot.axvline(x = Q_div_a_PN2,color='y',linestyle='dotted',linewidth=lw)
                lw+=0.5
            
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

    if len(Delta_is_FO)>0:
        indices_pos = [i for i in range(len(Delta_is_FO)) if Delta_is_FO[i] >= 0.0]
        indices_neg = [i for i in range(len(Delta_is_FO)) if Delta_is_FO[i] < 0.0]
        plot2.plot(plot_Q_div_a_points[indices_pos],Delta_is_FO[indices_pos]*180.0/np.pi,color='b',linestyle='dashed',linewidth=w)
        plot2.plot(plot_Q_div_a_points[indices_neg],-Delta_is_FO[indices_neg]*180.0/np.pi,color='r',linestyle='dashed',linewidth=w)

    if len(Delta_is_SO)>0:
        indices_pos = [i for i in range(len(Delta_is_SO)) if Delta_is_SO[i] >= 0.0]
        indices_neg = [i for i in range(len(Delta_is_SO)) if Delta_is_SO[i] < 0.0]
        plot2.plot(plot_Q_div_a_points[indices_pos],Delta_is_SO[indices_pos]*180.0/np.pi,color='b',linestyle='solid',linewidth=w)
        plot2.plot(plot_Q_div_a_points[indices_neg],-Delta_is_SO[indices_neg]*180.0/np.pi,color='r',linestyle='solid',linewidth=w)
        
    plot1.set_ylim(args.ymin,args.ymax)
    plot_c.set_ylim(args.ymin,args.ymax)
    
    plots = [plot1,plot2,plot_c]
    for plot in plots:
        plot.set_xlim(args.xmin,args.xmax)

    if args.show_inset == True:
        plot1_inset.set_xlim(7.5,8.5)
        plot1_inset.set_ylim(1.0e-4,4.0e-4)

    
    plots = [plot1,plot2]
    labels = [r"$\Delta e$",r"$\Delta i/\mathrm{deg}$"]
    for index,plot in enumerate(plots):
        if index==1:
            plot.set_xlabel(r"$Q/a$",fontsize=fontsize)
        plot.set_ylabel(labels[index],fontsize=fontsize)
                
        plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)

        if index in [0]:
            plot.set_xticklabels([])

    plot_c.set_xlabel(r"$Q/a$",fontsize=fontsize)
    plot_c.set_ylabel(labels[0],fontsize=fontsize)
    plot_c.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)

    for plot in [plot1,plot_c]:
        handles,labels = plot.get_legend_handles_labels()
        plot.legend(handles,labels,loc="upper right",fontsize=0.7*fontsize,framealpha=1)

        plot.set_title("$E = %s; m_1/m_2 = %s; \, m_1/M = %s; \, \,e=%s;\,i=%s\,\mathrm{deg};\, \omega=%s\,\mathrm{deg};\, \Omega=%s\,\mathrm{deg}$"%(args.e_per,round(args.m1/args.m2,1),round(args.m1/args.M_per),args.e,round(args.i*180.0/np.pi,1),round(args.omega*180.0/np.pi,1),round(args.Omega*180.0/np.pi,1)),fontsize=0.55*fontsize)    

    fig.subplots_adjust(hspace=0.0,wspace=0.0)
    filename = os.path.dirname(os.path.realpath(__file__)) + '/figs/Delta_es_is_' + str(args.name) + '_e_per_' + str(args.e_per) + '_e_' + str(args.e) + '_q_' + str(args.m1/args.m2) + '_i_' + str(args.i) + '_do_nbody_' + str(do_nbody) + '_Q_min_' + str(args.Q_min) + '_Q_max_' + str(args.Q_max) + '_N_Q_' + str(args.N_Q) + '_xmin_' + str(args.xmin) + '_xmax_' + str(args.xmax) + '_ymin_' + str(args.ymin) + '_ymax_' + str(args.ymax) + '.pdf'
    fig.savefig(filename,dpi=200)

    fig_c.subplots_adjust(hspace=0.0,wspace=0.0)
    filename = os.path.dirname(os.path.realpath(__file__)) + '/figs/Delta_es_' + str(args.name) + '_e_per_' + str(args.e_per) + '_e_' + str(args.e) + '_q_' + str(args.m1/args.m2)  + '_i_' + str(args.i) + '_do_nbody_' + str(do_nbody) + '_Q_min_' + str(args.Q_min) + '_Q_max_' + str(args.Q_max) + '_N_Q_' + str(args.N_Q) + '_xmin_' + str(args.xmin) + '_xmax_' + str(args.xmax) + '_ymin_' + str(args.ymin) + '_ymax_' + str(args.ymax) + '.pdf'
    fig_c.savefig(filename,dpi=200)
    
def plot_function_series_PN(args,data_series):
    Q_points = data_series["Q_points"]
    Delta_es_mean = data_series["Delta_es_mean"]
    Delta_is_mean = data_series["Delta_is_mean"]
    Delta_es_rms = data_series["Delta_es_rms"]
    Delta_is_rms = data_series["Delta_is_rms"]
    do_nbody = False
    
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
    
    ex,ey,ez,jx,jy,jz = core.orbital_elements_to_orbital_vectors(e,i,omega,Omega)
    
    Delta_es_FO,Delta_is_FO = [],[]
    Delta_es_SO,Delta_is_SO = [],[]
    Delta_es_TO,Delta_is_TO = [],[]
    
    plot_Q_div_a_points = pow(10.0,np.linspace(np.log10(np.amin(Q_div_a_points)),np.log10(np.amax(Q_div_a_points)),2000))
    
    Delta_es_TO_mean = []
    Delta_es_TO_rms = []
    
    g1=1.0
    g2=1.0
    for index,Q_div_a in enumerate(plot_Q_div_a_points):
        a_div_Q = 1.0/Q_div_a
        eps_SA = core.compute_eps_SA(m,M_per,a_div_Q,e_per)
        eps_oct = core.compute_eps_oct(m1,m2,m,a_div_Q,e_per)
        
        if args.include_analytic_FO_terms == True and 0==1:
            Delta_e_FO,Delta_i_FO = core.compute_FO_prediction(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz)

            Delta_es_FO.append(Delta_e_FO)
            Delta_is_FO.append(Delta_i_FO)    

            g1 = Delta_e_FO/eps_SA
            
        if args.include_analytic_SO_terms == True and 0==1:
            Delta_e_SO,Delta_i_SO = core.compute_SO_prediction(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz)
            
            Delta_es_SO.append(Delta_e_SO)
            Delta_is_SO.append(Delta_i_SO)
            
            g2 = (Delta_e_SO-Delta_e_FO)/(eps_SA**2)
            
        if args.include_analytic_TO_terms == True and 0==1:
            Delta_e_TO_only,Delta_i_TO_only = core.compute_TO_prediction_only(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz)

            Delta_e_TO = Delta_e_SO + Delta_e_TO_only
            Delta_i_TO = Delta_i_SO

            Delta_es_TO.append(Delta_e_TO)
            Delta_is_TO.append(Delta_i_TO)

        #Delta_es_TO_mean.append(  (1.0/(2.0*np.pi))*(9.0/256.)*e*np.pi**3*eps_SA**2*(124 - 299*e**2 + \
        #    4*(-56 + 81*e**2)*np.cos(2*i) + (36 + 39*e**2)*np.cos(4*i)    ) )
        Delta_e_TO_mean,Delta_e_TO_rms = core.compute_TO_prediction_averaged_over_argument_of_periapsis(eps_SA,eps_oct,e_per,e,i,Omega)
        Delta_es_TO_mean.append(Delta_e_TO_mean)
        Delta_es_TO_rms.append(Delta_e_TO_rms)

    
    a_div_Q_crit = pow( (-g1/g2)*(np.sqrt(m*(m+M_per))/M_per),2.0/3.0)*(1.0+e_per)
    Q_div_a_crit = 1.0/a_div_Q_crit

    Q_div_a_crit2 = Q_div_a_crit*pow(0.5,-2.0/3.0)


    Q_div_a_crit_R_unity = pow(0.5,-2.0/3.0)*pow( (1.0 + M_per/m)*(1.0 + e_per), 1.0/3.0)


    f_TB_omega = f_TB_omega_function(i,omega,Omega)
    #f_TB_omega = np.fabs( 4.0*(1.0 + 3.0*np.cos(2.0*i) + 6.0*np.cos(2.0*Omega)*np.sin(i)**2) )
    rg = args.G*args.m/(args.c**2)

    Delta_es_FO = np.array(Delta_es_FO)
    Delta_es_SO = np.array(Delta_es_SO)
    Delta_es_TO = np.array(Delta_es_TO)
    Delta_is_FO = np.array(Delta_is_FO)
    Delta_is_SO = np.array(Delta_is_SO)
    Delta_is_TO = np.array(Delta_is_TO)

    Delta_es_TO_mean = np.array(Delta_es_TO_mean)
    Delta_es_TO_rms = np.array(Delta_es_TO_rms)
    
    fontsize=args.fontsize
    labelsize=args.labelsize
    
    fig=pyplot.figure(figsize=(8,10))
    plot1=fig.add_subplot(2,1,1,yscale="log",xscale="log")
    plot2=fig.add_subplot(2,1,2,yscale="linear",xscale="log")
    fig_c=pyplot.figure(figsize=(8,6))
    plot_c=fig_c.add_subplot(1,1,1,yscale="log",xscale="log")
    

    if args.show_inset == True:
        from mpl_toolkits.axes_grid.inset_locator import inset_axes
        plot1_inset = inset_axes(plot_c,
        width="25%", # width = 30% of parent_bbox
        height=1.4, # height : 1 inch
        loc="lower right")
        plot1_inset.set_xscale("log")
        plot1_inset.set_yscale("log")

    s_nbody=50
    s=10
    if args.show_inset == True:
        plots = [plot1,plot_c,plot1_inset]
    else:
        plots = [plot1,plot_c]
    for plot in plots:
        #plot.axvline(x=Q_div_a_crit,color='k',linestyle='dotted')
        #plot.axvline(x=Q_div_a_crit2,color='k',linestyle='dotted')
        plot.axvline(x=Q_div_a_crit_R_unity,color='g',linestyle='dotted')
    
        indices_pos = [i for i in range(len(Delta_es_mean)) if Delta_es_mean[i] >= 0.0]
        indices_neg = [i for i in range(len(Delta_es_mean)) if Delta_es_mean[i] < 0.0]
        plot.scatter(Q_div_a_points[indices_pos],Delta_es_mean[indices_pos],color='b',s=s, facecolors='none',label="$\mathrm{SA\,mean}$")
        plot.scatter(Q_div_a_points[indices_neg],-Delta_es_mean[indices_neg],color='r',s=s, facecolors='none')

        indices_pos = [i for i in range(len(Delta_es_rms)) if Delta_es_rms[i] >= 0.0]
        indices_neg = [i for i in range(len(Delta_es_rms)) if Delta_es_rms[i] < 0.0]
        plot.scatter(Q_div_a_points[indices_pos],Delta_es_rms[indices_pos],color='b',s=2*s, facecolors='none',label="$\mathrm{SA\,rms}$",marker='*')
        plot.scatter(Q_div_a_points[indices_neg],-Delta_es_rms[indices_neg],color='r',s=2*s, facecolors='none',marker='*')

        if do_nbody == True:
            s=s_nbody
            indices_pos = [i for i in range(len(nbody_Delta_es)) if nbody_Delta_es[i] >= 0.0]
            indices_neg = [i for i in range(len(nbody_Delta_es)) if nbody_Delta_es[i] < 0.0]
            plot.scatter(Q_div_a_points[indices_pos],nbody_Delta_es[indices_pos],color='b',s=s, facecolors='none',label="$\mathrm{3-body}$",marker='*')
            plot.scatter(Q_div_a_points[indices_neg],-nbody_Delta_es[indices_neg],color='r',s=s, facecolors='none',marker='*')

        w=2.0
        #plot.axhline(y = 1.0 - e,color='k',linestyle='dotted',linewidth=w,label="$1-e$")
        
        #if len(Delta_es_FO)>0:
        if 1==0:
            indices_pos = [i for i in range(len(Delta_es_FO)) if Delta_es_FO[i] >= 0.0]
            indices_neg = [i for i in range(len(Delta_es_FO)) if Delta_es_FO[i] < 0.0]
            plot.plot(plot_Q_div_a_points[indices_pos],Delta_es_FO[indices_pos],color='b',linestyle='dashed',linewidth=w,label="$\mathrm{FO}$")
            plot.plot(plot_Q_div_a_points[indices_neg],-Delta_es_FO[indices_neg],color='r',linestyle='dashed',linewidth=w)

        if len(Delta_es_SO)>0:
            indices_pos = [i for i in range(len(Delta_es_SO)) if Delta_es_SO[i] >= 0.0]
            indices_neg = [i for i in range(len(Delta_es_SO)) if Delta_es_SO[i] < 0.0]
            plot.plot(plot_Q_div_a_points[indices_pos],Delta_es_SO[indices_pos],color='b',linestyle='solid',linewidth=w,label="$\mathrm{SO}$")
            plot.plot(plot_Q_div_a_points[indices_neg],-Delta_es_SO[indices_neg],color='r',linestyle='solid',linewidth=w)    

        if 1==0:
        #if len(Delta_es_TO)>0:
            indices_pos = [i for i in range(len(Delta_es_TO)) if Delta_es_TO[i] >= 0.0]
            indices_neg = [i for i in range(len(Delta_es_TO)) if Delta_es_TO[i] < 0.0]
            plot.plot(plot_Q_div_a_points[indices_pos],Delta_es_TO[indices_pos],color='b',linestyle='dotted',linewidth=w,label="$\mathrm{TO}$")
            plot.plot(plot_Q_div_a_points[indices_neg],-Delta_es_TO[indices_neg],color='r',linestyle='dotted',linewidth=w)    
        
        if len(Delta_es_TO_mean)>0:
            indices_pos = [i for i in range(len(Delta_es_TO_mean)) if Delta_es_TO_mean[i] >= 0.0]
            indices_neg = [i for i in range(len(Delta_es_TO_mean)) if Delta_es_TO_mean[i] < 0.0]
            plot.plot(plot_Q_div_a_points[indices_pos],Delta_es_TO_mean[indices_pos],color='b',linestyle='solid',linewidth=w,label="$\mathrm{TO\,mean}$")
            plot.plot(plot_Q_div_a_points[indices_neg],-Delta_es_TO_mean[indices_neg],color='r',linestyle='solid',linewidth=w)            

        if len(Delta_es_TO_rms)>0:
            indices_pos = [i for i in range(len(Delta_es_TO_rms)) if Delta_es_TO_rms[i] >= 0.0]
            indices_neg = [i for i in range(len(Delta_es_TO_rms)) if Delta_es_TO_rms[i] < 0.0]
            plot.plot(plot_Q_div_a_points[indices_pos],Delta_es_TO_rms[indices_pos],color='b',linestyle='dashed',linewidth=w,label="$\mathrm{TO\,rms}$")
            plot.plot(plot_Q_div_a_points[indices_neg],-Delta_es_TO_rms[indices_neg],color='r',linestyle='dashed',linewidth=w)            

        if 1==1:
            alphas = [0.1,0.5,1.0,2.0,10.0]
            lw=0.5
            for index_alpha,alpha in enumerate(alphas):
                Q_div_a_PN = np.sqrt(1.0-args.e**2)*pow( (1.0/64.0)*np.fabs(f_TB_omega)*alpha*(args.a/rg)*(args.M_per/args.m), 1.0/3.0)
                if index_alpha==0:
                    Q_div_a_PN2 = pow( (1.0-args.e**2)*(1.0/3.0)*(args.a/rg)*(args.M_per/args.m),1.0/3.0)
                    plot.axvline(x = Q_div_a_PN2,color='y',linestyle='dashed',linewidth=w)

                    Q_div_a_PN3 = pow( (1.0/3.0)*(1.0-args.e**2)*(args.a/rg)*np.sqrt((args.m+args.M_per)/args.m),2.0/3.0)
                    plot.axvline(x = Q_div_a_PN3,color='k',linestyle='dotted',linewidth=w)
                
                lw+=0.5
            
    if do_nbody == True:
        s=s_nbody
        indices_pos = [i for i in range(len(nbody_Delta_is)) if nbody_Delta_is[i] >= 0.0]
        indices_neg = [i for i in range(len(nbody_Delta_is)) if nbody_Delta_is[i] < 0.0]
        plot2.scatter(Q_div_a_points[indices_pos],nbody_Delta_is[indices_pos]*180.0/np.pi,color='b',s=s, facecolors='none',marker='*')
        plot2.scatter(Q_div_a_points[indices_neg],-nbody_Delta_is[indices_neg]*180.0/np.pi,color='r',s=s, facecolors='none',marker='*')



    indices_pos = [i for i in range(len(Delta_is_mean)) if Delta_is_mean[i] >= 0.0]
    indices_neg = [i for i in range(len(Delta_is_mean)) if Delta_is_mean[i] < 0.0]
    plot2.scatter(Q_div_a_points[indices_pos],Delta_is_mean[indices_pos]*180.0/np.pi,color='b',s=s, facecolors='none')
    plot2.scatter(Q_div_a_points[indices_neg],-Delta_is_mean[indices_neg]*180.0/np.pi,color='r',s=s, facecolors='none')

    if len(Delta_is_FO)>0:
        indices_pos = [i for i in range(len(Delta_is_FO)) if Delta_is_FO[i] >= 0.0]
        indices_neg = [i for i in range(len(Delta_is_FO)) if Delta_is_FO[i] < 0.0]
        plot2.plot(plot_Q_div_a_points[indices_pos],Delta_is_FO[indices_pos]*180.0/np.pi,color='b',linestyle='dashed',linewidth=w)
        plot2.plot(plot_Q_div_a_points[indices_neg],-Delta_is_FO[indices_neg]*180.0/np.pi,color='r',linestyle='dashed',linewidth=w)

    if len(Delta_is_SO)>0:
        indices_pos = [i for i in range(len(Delta_is_SO)) if Delta_is_SO[i] >= 0.0]
        indices_neg = [i for i in range(len(Delta_is_SO)) if Delta_is_SO[i] < 0.0]
        plot2.plot(plot_Q_div_a_points[indices_pos],Delta_is_SO[indices_pos]*180.0/np.pi,color='b',linestyle='solid',linewidth=w)
        plot2.plot(plot_Q_div_a_points[indices_neg],-Delta_is_SO[indices_neg]*180.0/np.pi,color='r',linestyle='solid',linewidth=w)
        
    plot1.set_ylim(args.ymin,args.ymax)
    plot_c.set_ylim(args.ymin,args.ymax)
    
    plots = [plot1,plot2,plot_c]
    for plot in plots:
        plot.set_xlim(args.xmin,args.xmax)

    if args.show_inset == True:
        plot1_inset.set_xlim(7.5,8.5)
        plot1_inset.set_ylim(1.0e-4,4.0e-4)

    
    if args.include_1PN_terms == True:
        plot_c.annotate("$\mathrm{1PN\,terms\,included}$",xy=(0.1,0.9),xycoords='axes fraction',fontsize=0.7*fontsize)
    if args.include_1PN_terms == False:
        plot_c.annotate("$\mathrm{1PN\,terms\,excluded}$",xy=(0.1,0.9),xycoords='axes fraction',fontsize=0.7*fontsize)
    
    plots = [plot1,plot2]
    labels = [r"$\Delta e$",r"$\Delta i/\mathrm{deg}$"]
    for index,plot in enumerate(plots):
        if index==1:
            plot.set_xlabel(r"$Q/a$",fontsize=fontsize)
        plot.set_ylabel(labels[index],fontsize=fontsize)
                
        plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)

        if index in [0]:
            plot.set_xticklabels([])

    plot_c.set_xlabel(r"$Q/a$",fontsize=fontsize)
    plot_c.set_ylabel(labels[0],fontsize=fontsize)
    plot_c.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)

    for plot in [plot1,plot_c]:
        handles,labels = plot.get_legend_handles_labels()
        plot.legend(handles,labels,loc="upper right",fontsize=0.7*fontsize,framealpha=1)

        plot.set_title("$E = %s; m_1/m_2 = %s; \, m_1/M = %s; \, \,e=%s;\,i=%s\,\mathrm{deg};\, \Omega=%s\,\mathrm{deg}$"%(args.e_per,round(args.m1/args.m2,1),round(args.m1/args.M_per),args.e,round(args.i*180.0/np.pi,1),round(args.Omega*180.0/np.pi,1)),fontsize=0.55*fontsize)    

    fig.subplots_adjust(hspace=0.0,wspace=0.0)
    filename = os.path.dirname(os.path.realpath(__file__)) + '/figs/PN_Delta_es_is_' + str(args.name) + '_e_per_' + str(args.e_per) + '_e_' + str(args.e) + '_q_' + str(args.m1/args.m2) + '_i_' + str(args.i) + '_do_nbody_' + str(do_nbody) + '_Q_min_' + str(args.Q_min) + '_Q_max_' + str(args.Q_max) + '_N_Q_' + str(args.N_Q) + '_xmin_' + str(args.xmin) + '_xmax_' + str(args.xmax) + '_ymin_' + str(args.ymin) + '_ymax_' + str(args.ymax) + '_include_1PN_terms_' + str(args.include_1PN_terms) + '_fraction_theta_0_' + str(args.fraction_theta_0) + '_N_AP_' + str(args.N_AP) + '_a_' + str(args.a) + '.pdf'
    fig.savefig(filename,dpi=200)

    fig_c.subplots_adjust(hspace=0.0,wspace=0.0)
    filename = os.path.dirname(os.path.realpath(__file__)) + '/figs/PN_Delta_es_' + str(args.name) + '_e_per_' + str(args.e_per) + '_e_' + str(args.e) + '_q_' + str(args.m1/args.m2)  + '_i_' + str(args.i) + '_do_nbody_' + str(do_nbody) + '_Q_min_' + str(args.Q_min) + '_Q_max_' + str(args.Q_max) + '_N_Q_' + str(args.N_Q) + '_xmin_' + str(args.xmin) + '_xmax_' + str(args.xmax) + '_ymin_' + str(args.ymin) + '_ymax_' + str(args.ymax) + '_include_1PN_terms_' + str(args.include_1PN_terms) + '_fraction_theta_0_' + str(args.fraction_theta_0) + '_N_AP_' + str(args.N_AP) + '_a_' + str(args.a) + '.pdf'
    fig_c.savefig(filename,dpi=200)


def plot_function_series_angles(args,data_series_angles):
    i_points = data_series_angles["i_points"]
    AP_points = data_series_angles["AP_points"]
    LAN_points = data_series_angles["LAN_points"]

    Delta_es_i = data_series_angles["Delta_es_i"]
    Delta_is_i = data_series_angles["Delta_is_i"]
    do_nbody = data_series_angles["do_nbody"]
    nbody_Delta_es_i = data_series_angles["nbody_Delta_es_i"]
    nbody_Delta_is_i = data_series_angles["nbody_Delta_is_i"]

    Delta_es_AP = data_series_angles["Delta_es_AP"]
    Delta_is_AP = data_series_angles["Delta_is_AP"]
    do_nbody = data_series_angles["do_nbody"]
    nbody_Delta_es_AP = data_series_angles["nbody_Delta_es_AP"]
    nbody_Delta_is_AP = data_series_angles["nbody_Delta_is_AP"]

    Delta_es_LAN = data_series_angles["Delta_es_LAN"]
    Delta_is_LAN = data_series_angles["Delta_is_LAN"]
    do_nbody = data_series_angles["do_nbody"]
    nbody_Delta_es_LAN = data_series_angles["nbody_Delta_es_LAN"]
    nbody_Delta_is_LAN = data_series_angles["nbody_Delta_is_LAN"]

    m = args.m
    M_per = args.M_per
    m1 = args.m1
    m2 = args.m2
    e_per = args.e_per
    a = args.a
    e = args.e
    
    Q = args.Q

    i0 = args.i
    AP0 = args.omega
    LAN0 = args.Omega
    print( 'i0',i0,'AP0',AP0,'LAN0',LAN0)
    data_points_all = [i_points,AP_points,LAN_points]
    N_p = 2000
    plot_points_all = [np.linspace(0.0,np.pi,N_p),np.linspace(0.0,2.0*np.pi,N_p),np.linspace(0.0,2.0*np.pi,N_p)]
    Delta_es_plots_all = [Delta_es_i,Delta_es_AP,Delta_es_LAN]
    Delta_is_plots_all = [Delta_is_i,Delta_is_AP,Delta_is_LAN]
    nbody_Delta_es_plots_all = [nbody_Delta_es_i,nbody_Delta_es_AP,nbody_Delta_es_LAN]
    nbody_Delta_is_plots_all = [nbody_Delta_is_i,nbody_Delta_is_AP,nbody_Delta_is_LAN]

    fontsize=args.fontsize
    labelsize=args.labelsize

    angle_descriptions = ["i","AP","LAN"]
    angle_description_labels = [r"$i/\mathrm{deg}$",r"$\omega/\mathrm{deg}$",r"$\Omega/\mathrm{deg}$"]
    
    rad_to_deg = 180.0/np.pi
    
    for index_points,plot_points in enumerate(plot_points_all):
        angle_description = angle_descriptions[index_points]
        data_points = data_points_all[index_points]
        
        if index_points == 0:
            title = "$E = %s; m_1/m_2 = %s; \, m_1/M = %s; \, e=%s;\,Q/a=%s;\, \omega=%s\,\mathrm{deg};\, \Omega=%s\,\mathrm{deg}$"%(args.e_per,round(args.m1/args.m2,1),round(args.m1/args.M_per),args.e,round(Q/a,1),round(AP0*180.0/np.pi,1),round(LAN0*180.0/np.pi,1))
        elif index_points == 1:
            title = "$E = %s; m_1/m_2 = %s; \, m_1/M = %s; \,e=%s;\,Q/a=%s;\, i=%s\,\mathrm{deg};\,\mathrm{deg};\, \Omega=%s\,\mathrm{deg}$"%(args.e_per,round(args.m1/args.m2,1),round(args.m1/args.M_per),args.e,round(Q/a,1),round(i0*180.0/np.pi,1),round(LAN0*180.0/np.pi,1))
        elif index_points == 2:
            title = "$E = %s; m_1/m_2 = %s; \, m_1/M = %s; \,e=%s;\,Q/a=%s;\, i=%s\,\mathrm{deg};\,\mathrm{deg};\, \omega=%s\,\mathrm{deg}$"%(args.e_per,round(args.m1/args.m2,1),round(args.m1/args.M_per),args.e,round(Q/a,1),round(i0*180.0/np.pi,1),round(AP0*180.0/np.pi,1))
        
        Delta_es = Delta_es_plots_all[index_points]
        Delta_is = Delta_is_plots_all[index_points]
        nbody_Delta_es = nbody_Delta_es_plots_all[index_points]
        nbody_Delta_is = nbody_Delta_is_plots_all[index_points]
        
        Delta_es_FO,Delta_is_FO = [],[]
        Delta_es_SO,Delta_is_SO = [],[]
        Delta_es_TO,Delta_is_TO = [],[]
    
        for j,point in enumerate(plot_points):
            if index_points==0:
                i = point
                omega = AP0
                Omega = LAN0
            elif index_points==1:
                i = i0
                omega = point
                Omega = LAN0
            elif index_points==2:
                i = i0
                omega = AP0
                Omega = point
                
            ex,ey,ez,jx,jy,jz = core.orbital_elements_to_orbital_vectors(e,i,omega,Omega)
        
            a_div_Q = a/Q
            eps_SA = core.compute_eps_SA(m,M_per,a_div_Q,e_per)
            eps_oct = core.compute_eps_oct(m1,m2,m,a_div_Q,e_per)
        
            if args.include_analytic_FO_terms == True:
                Delta_e_FO,Delta_i_FO = core.compute_FO_prediction(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz)

                Delta_es_FO.append(Delta_e_FO)
                Delta_is_FO.append(Delta_i_FO)    

            if args.include_analytic_SO_terms == True:
                Delta_e_SO,Delta_i_SO = core.compute_SO_prediction(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz)
                
                Delta_es_SO.append(Delta_e_SO)
                Delta_is_SO.append(Delta_i_SO)
                
            if args.include_analytic_TO_terms == True:
                Delta_e_TO_only,Delta_i_TO_only = core.compute_TO_prediction_only(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz)

                Delta_e_TO = Delta_e_SO + Delta_e_TO_only
                Delta_i_TO = Delta_i_SO

                Delta_es_TO.append(Delta_e_TO)
                Delta_is_TO.append(Delta_i_TO)

        
            g1 = Delta_e_FO/eps_SA
            g2 = (Delta_e_SO-Delta_e_FO)/(eps_SA**2)

        Delta_es_FO = np.array(Delta_es_FO)
        Delta_es_SO = np.array(Delta_es_SO)
        Delta_es_TO = np.array(Delta_es_TO)
        Delta_is_FO = np.array(Delta_is_FO)
        Delta_is_SO = np.array(Delta_is_SO)
        Delta_is_TO = np.array(Delta_is_TO)

    
        fig=pyplot.figure(figsize=(8,10))
        plot1=fig.add_subplot(2,1,1,yscale="log",xscale="linear")
        plot2=fig.add_subplot(2,1,2,yscale="linear",xscale="linear")
        fig_c=pyplot.figure(figsize=(8,6))
        plot_c=fig_c.add_subplot(1,1,1,yscale="log",xscale="linear")
    

        s_nbody=50
        s=10
        plots = [plot1,plot_c]
        for plot in plots:
                    
            indices_pos = [i for i in range(len(Delta_es)) if Delta_es[i] >= 0.0]
            indices_neg = [i for i in range(len(Delta_es)) if Delta_es[i] < 0.0]
            plot.scatter(rad_to_deg*data_points[indices_pos],Delta_es[indices_pos],color='b',s=s, facecolors='none',label="$\mathrm{SA}$")
            plot.scatter(rad_to_deg*data_points[indices_neg],-Delta_es[indices_neg],color='r',s=s, facecolors='none')

            if do_nbody == True:
                s=s_nbody
                indices_pos = [i for i in range(len(nbody_Delta_es)) if nbody_Delta_es[i] >= 0.0]
                indices_neg = [i for i in range(len(nbody_Delta_es)) if nbody_Delta_es[i] < 0.0]
                plot.scatter(rad_to_deg*data_points[indices_pos],nbody_Delta_es[indices_pos],color='b',s=s, facecolors='none',label="$\mathrm{3-body}$",marker='*')
                plot.scatter(rad_to_deg*data_points[indices_neg],-nbody_Delta_es[indices_neg],color='r',s=s, facecolors='none',marker='*')

            w=2.0
            
            if len(Delta_es_FO)>0:
                indices_pos = [i for i in range(len(Delta_es_FO)) if Delta_es_FO[i] >= 0.0]
                indices_neg = [i for i in range(len(Delta_es_FO)) if Delta_es_FO[i] < 0.0]
                plot.plot(rad_to_deg*plot_points[indices_pos],Delta_es_FO[indices_pos],color='b',linestyle='dashed',linewidth=w,label="$\mathrm{FO}$")
                plot.plot(rad_to_deg*plot_points[indices_neg],-Delta_es_FO[indices_neg],color='r',linestyle='dashed',linewidth=w)

            if len(Delta_es_SO)>0:
                indices_pos = [i for i in range(len(Delta_es_SO)) if Delta_es_SO[i] >= 0.0]
                indices_neg = [i for i in range(len(Delta_es_SO)) if Delta_es_SO[i] < 0.0]
                plot.plot(rad_to_deg*plot_points[indices_pos],Delta_es_SO[indices_pos],color='b',linestyle='solid',linewidth=w,label="$\mathrm{SO}$")
                plot.plot(rad_to_deg*plot_points[indices_neg],-Delta_es_SO[indices_neg],color='r',linestyle='solid',linewidth=w)    

            if len(Delta_es_TO)>0:
                indices_pos = [i for i in range(len(Delta_es_TO)) if Delta_es_TO[i] >= 0.0]
                indices_neg = [i for i in range(len(Delta_es_TO)) if Delta_es_TO[i] < 0.0]
                plot.plot(rad_to_deg*plot_points[indices_pos],Delta_es_TO[indices_pos],color='b',linestyle='dotted',linewidth=w,label="$\mathrm{TO}$")
                plot.plot(rad_to_deg*plot_points[indices_neg],-Delta_es_TO[indices_neg],color='r',linestyle='dotted',linewidth=w)    
                
        if do_nbody == True:
            s=s_nbody
            indices_pos = [i for i in range(len(nbody_Delta_is)) if nbody_Delta_is[i] >= 0.0]
            indices_neg = [i for i in range(len(nbody_Delta_is)) if nbody_Delta_is[i] < 0.0]
            plot2.scatter(rad_to_deg*data_points[indices_pos],nbody_Delta_is[indices_pos]*180.0/np.pi,color='b',s=s, facecolors='none',marker='*')
            plot2.scatter(rad_to_deg*data_points[indices_neg],-nbody_Delta_is[indices_neg]*180.0/np.pi,color='r',s=s, facecolors='none',marker='*')


        indices_pos = [i for i in range(len(Delta_is)) if Delta_is[i] >= 0.0]
        indices_neg = [i for i in range(len(Delta_is)) if Delta_is[i] < 0.0]
        plot2.scatter(rad_to_deg*data_points[indices_pos],Delta_is[indices_pos]*180.0/np.pi,color='b',s=s, facecolors='none')
        plot2.scatter(rad_to_deg*data_points[indices_neg],-Delta_is[indices_neg]*180.0/np.pi,color='r',s=s, facecolors='none')

        if len(Delta_is_FO)>0:
            indices_pos = [i for i in range(len(Delta_is_FO)) if Delta_is_FO[i] >= 0.0]
            indices_neg = [i for i in range(len(Delta_is_FO)) if Delta_is_FO[i] < 0.0]
            plot2.plot(rad_to_deg*plot_points[indices_pos],Delta_is_FO[indices_pos]*180.0/np.pi,color='b',linestyle='dashed',linewidth=w)
            plot2.plot(rad_to_deg*plot_points[indices_neg],-Delta_is_FO[indices_neg]*180.0/np.pi,color='r',linestyle='dashed',linewidth=w)

        if len(Delta_is_SO)>0:
            indices_pos = [i for i in range(len(Delta_is_SO)) if Delta_is_SO[i] >= 0.0]
            indices_neg = [i for i in range(len(Delta_is_SO)) if Delta_is_SO[i] < 0.0]
            plot2.plot(rad_to_deg*plot_points[indices_pos],Delta_is_SO[indices_pos]*180.0/np.pi,color='b',linestyle='solid',linewidth=w)
            plot2.plot(rad_to_deg*plot_points[indices_neg],-Delta_is_SO[indices_neg]*180.0/np.pi,color='r',linestyle='solid',linewidth=w)
            
        plot1.set_ylim(args.ymin,args.ymax)
        plot_c.set_ylim(args.ymin,args.ymax)

        
        plots = [plot1,plot2]
        labels = [r"$\Delta e$",r"$\Delta i/\mathrm{deg}$"]
        for index,plot in enumerate(plots):
            plot.set_ylabel(labels[index],fontsize=fontsize)
                    
            plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)

            if index in [0]:
                plot.set_xticklabels([])

        plot_c.set_xlabel(angle_description_labels[index_points],fontsize=fontsize)
        plot_c.set_ylabel(labels[0],fontsize=fontsize)
        plot_c.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)

        for plot in [plot1,plot_c]:
            handles,labels = plot.get_legend_handles_labels()
            plot.legend(handles,labels,loc="upper right",fontsize=0.7*fontsize,framealpha=1)

            plot.set_title(title,fontsize=0.55*fontsize)    
        
        
        fig.subplots_adjust(hspace=0.0,wspace=0.0)
        filename = os.path.dirname(os.path.realpath(__file__)) + '/figs/series_' + str(angle_description) + '_Delta_es_is_' + str(args.name) + '_e_per_' + str(args.e_per) + '_e_' + str(args.e) + '_q_' + str(args.m1/args.m2) + '_i_' + str(args.i) + '_do_nbody_' + str(do_nbody) + '_Q_min_' + str(args.Q_min) + '_Q_max_' + str(args.Q_max) + '_N_Q_' + str(args.N_Q) + '_xmin_' + str(args.xmin) + '_xmax_' + str(args.xmax) + '_ymin_' + str(args.ymin) + '_ymax_' + str(args.ymax) + '.pdf'
        fig.savefig(filename,dpi=200)

        fig_c.subplots_adjust(hspace=0.0,wspace=0.0)
        filename = os.path.dirname(os.path.realpath(__file__)) + '/figs/series_' + str(angle_description) + '_Delta_es_' + str(args.name) + '_e_per_' + str(args.e_per) + '_e_' + str(args.e) + '_q_' + str(args.m1/args.m2)  + '_i_' + str(args.i) + '_do_nbody_' + str(do_nbody) + '_Q_min_' + str(args.Q_min) + '_Q_max_' + str(args.Q_max) + '_N_Q_' + str(args.N_Q) + '_xmin_' + str(args.xmin) + '_xmax_' + str(args.xmax) + '_ymin_' + str(args.ymin) + '_ymax_' + str(args.ymax) + '.pdf'
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
    
    ex,ey,ez,jx,jy,jz = core.orbital_elements_to_orbital_vectors(e,i,omega,Omega)
        
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
        eps_oct = 0.0
        
        Delta_e_FO,Delta_i_FO = compute_FO_prediction(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz)


        label = r"$Q/a\simeq%s$"%(round(Q_div_a,1))
        color=colors[index_plot]
        linewidth=linewidths[index_plot]
        linestyle=linestyles[index_plot]
        plot1.plot(thetas*180.0/np.pi,es,color=color,linestyle=linestyle,linewidth=linewidth,label=label)
        plot2.plot(thetas*180.0/np.pi,np.array(incls)*180.0/np.pi,color=color,linestyle=linestyle,linewidth=linewidth,label=label)

        plot1.axhline(y=e+Delta_e_FO,linestyle=linestyle,linewidth=linewidth,color='r')
        
        index_plot+=1    
    
    plots=[plot1,plot2]
    labels = [r"$e$",r"$i/\mathrm{deg}$"]
    for index,plot in enumerate(plots):
        if index==1:
            plot.set_xlabel(r"$\theta/\mathrm{deg}$",fontsize=fontsize)
        plot.set_ylabel(labels[index],fontsize=fontsize)
                
        plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)

        if index in [0]:
            plot.set_xticklabels([])


    handles,labels = plot1.get_legend_handles_labels()
    plot1.legend(handles,labels,loc="upper left",fontsize=0.6*fontsize)

    plot1.set_title("$E = %s; m_1/m_2 = %s; \, \,e=%s;\,i=%s\,\mathrm{deg};\, \omega=%s\,\mathrm{deg};\, \Omega=%s\,\mathrm{deg}$"%(args.e_per,round(args.m1/args.m2,1),args.e,round(args.i*180.0/np.pi,1),round(args.omega*180.0/np.pi,1),round(args.Omega*180.0/np.pi,1)),fontsize=0.65*fontsize)    
    
    
    fig.subplots_adjust(hspace=0.0,wspace=0.0)
    filename = os.path.dirname(os.path.realpath(__file__)) + '/figs/elements_time_series_' + str(args.name) + '_e_per_' + str(args.e_per) + '_e_' + str(args.e) + '_q_' + str(args.m1/args.m2) + '_Q_min_' + str(args.Q_min) + '_Q_max_' + str(args.Q_max) + '_N_Q_' + str(args.N_Q) + '.pdf'
    fig.savefig(filename,dpi=200)

def f_TB_omega_function(i,omega,Omega):
    Cos = np.cos
    
    return (4 + 12*Cos(2*i) - 10*Cos(2*(i - omega)) + 20*Cos(2*omega) - 10*Cos(2*(i + omega)) - 20*Cos(i + 2*omega - 2*Omega) - 6*Cos(2*(i - Omega)) + 5*Cos(2*(i - omega - Omega)) + 30*Cos(2*(omega - Omega)) + 5*Cos(2*(i + omega - Omega)) + 12*Cos(2*Omega) - 6*Cos(2*(i + Omega)) + 5*Cos(2*(i - omega + Omega)) + 30*Cos(2*(omega + Omega)) + 5*Cos(2*(i + omega + Omega)) - 20*Cos(i - 2*omega + 2*Omega) + 20*Cos(i - 2*(omega + Omega)) + 20*Cos(i + 2*(omega + Omega)))

def f_TB_e_function(i,omega,Omega):
    Cos = np.cos
    Sin = np.sin
    
    return (((3 + Cos(2*i))*Cos(2*Omega) + 2*Sin(i)**2)*Sin(2*omega) + 4*Cos(i)*Cos(2*omega)*Sin(2*Omega))
    
def plot_function_overview(args):
    m = args.m
    M_per = args.M_per
    m1 = args.m1
    m2 = args.m2
    e_per = args.e_per
    a = args.a
    e = args.e
    

    i = args.i
    omega = args.omega
    Omega = args.Omega
    #print 'test0',f_TB_omega_function(i,omega,Omega),f_TB_e_function(i,omega,Omega)

    f_TB_omega = f_TB_omega_function(i,omega,Omega)
    f_TB_e = f_TB_e_function(i,omega,Omega)
        
    N=1000
    e_points = 1.0 - pow(10.0,np.linspace(-6.0,-0.0,N))
    
    e_per_values = [1.0,10.0]
    m1_values = [10.0,20.0]
    m2_values = [10.0,20.0]
    M_per_values = [10.0,20.0]
    a_values = [0.1,1.0]
    linestyles = ['solid','dashed']
    
    plot_Q_div_a_points = pow(10.0,np.linspace(0.0,np.log10(30.0),N))

    CONST_G = args.G
    CONST_C = args.c
    fontsize=args.fontsize
    labelsize=args.labelsize
    
    for index_e_per, e_per in enumerate(e_per_values):


        Q_div_a_points_sign_change = []
        Q_div_a_points_sign_plateau = []
        
        Q_div_a_points_ome = []

        
        for index_e,e in enumerate(e_points):
            eps_SA = 1.0 ### arbitrary since it is divided out below to obtain the f & g's
            eps_oct = 0.0

            ex,ey,ez,jx,jy,jz = core.orbital_elements_to_orbital_vectors(e,i,omega,Omega)        
            
            if args.include_analytic_FO_terms == True:
                Delta_e_FO,Delta_i_FO = core.compute_FO_prediction(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz)

            if args.include_analytic_SO_terms == True:
                Delta_e_SO,Delta_i_SO = core.compute_SO_prediction(args,eps_SA,eps_oct,e_per,ex,ey,ez,jx,jy,jz)
				            
            f = Delta_e_FO/eps_SA
            g = (Delta_e_SO-Delta_e_FO)/(eps_SA**2)
        
            eps_SA_sign_change = -(f/g)
            Q_div_a_sign_change = core.compute_Q_div_a_from_eps_SA(eps_SA_sign_change,m,M_per,e_per)
            Q_div_a_sign_plateau = pow(2.0,2.0/3.0)*Q_div_a_sign_change
            
            eps_SA_ome = (-f + np.sqrt(f**2 + 4.0*g*(1.0-e)))/(2.0*g)
            Q_div_a_ome = core.compute_Q_div_a_from_eps_SA(eps_SA_ome,m,M_per,e_per)
            
            Q_div_a_points_sign_change.append(Q_div_a_sign_change)
            Q_div_a_points_sign_plateau.append(Q_div_a_sign_plateau)
            Q_div_a_points_ome.append(Q_div_a_ome)

        Q_div_a_points_sign_change = np.array(Q_div_a_points_sign_change)
        Q_div_a_points_sign_plateau = np.array(Q_div_a_points_sign_plateau)
        Q_div_a_points_ome = np.array(Q_div_a_points_ome)
        
        Q_div_a_crit_R_unity = pow(0.5,-2.0/3.0)*pow( (1.0 + M_per/m)*(1.0 + e_per), 1.0/3.0)
        
        fig=pyplot.figure(figsize=(8,6))
        plot=fig.add_subplot(1,1,1,yscale="log",xscale="log")

        w=2.0
        plot.plot(Q_div_a_points_sign_change,1.0-e_points,color='k',linewidth=1.5*w,label="$\Delta e=0$",zorder=10,linestyle='dotted')
        plot.plot(Q_div_a_points_sign_plateau,1.0-e_points,color='k',linestyle='solid',linewidth=1.5*w,label="$\mathrm{Plateau}$",zorder=10)
        plot.axvline(x=Q_div_a_crit_R_unity,color='g',linestyle='dotted',linewidth=w)

        w_values = [0.5,2.0]
        for index_m in range(len(m1_values)):
            w = w_values[index_m]
            m1 = m1_values[index_m]
            m2 = m2_values[index_m]
            M_per = M_per_values[index_m]
            m=m1+m2    

            for index_system in range(len(a_values)):
                a = a_values[index_system]
                label1PN = "$a = %s\,\mathrm{AU\,(1PN)}$"%round(a,1)
                label25PN = "$a = %s\,\mathrm{AU\,(2.5PN)}$"%round(a,1)
                linestyle=linestyles[index_system]
                
                e_points_1PN = []
                e_points_1PNb = []
                e_points_25PN = []
                for index_Q_div_a,Q_div_a in enumerate(plot_Q_div_a_points):
                    rg = CONST_G*m/(CONST_C**2)
                    x = (64.0/np.fabs(f_TB_omega))*(rg/a)*Q_div_a**3*(m/M_per)

                    e_1PN = np.sqrt(1.0 - pow(x,2.0/3.0))
                    
                    x = (272.0/(9.0*np.fabs(f_TB_e)))*(m/M_per)*(m1*m2/(m**2))*Q_div_a**3*pow(rg/a,5.0/2.0)
                    e_25PN = np.sqrt(1.0 - pow(x,1.0/3.0))
                   
                    xb = (rg/a)*pow(Q_div_a,3.0/2.0)*np.sqrt(m/(m+M_per))

                    e_1PNb = np.sqrt(1.0 - xb)

                    e_points_1PN.append(e_1PN)
                    e_points_1PNb.append(e_1PNb)
                    e_points_25PN.append(e_25PN)
                    
                e_points_1PN = np.array(e_points_1PN)
                e_points_1PNb = np.array(e_points_1PNb)
                e_points_25PN = np.array(e_points_25PN)
                
                plot.plot(plot_Q_div_a_points,1.0-e_points_1PN,color='r',linestyle=linestyle,linewidth=1.5*w,label=label1PN)
                plot.plot(plot_Q_div_a_points,1.0-e_points_25PN,color='b',linestyle=linestyle,linewidth=0.75*w,label=label25PN)
                
        plot.set_xlim(1.0,25.0)
        plot.set_ylim(1e-6,1e0)
        
        plot.set_xlabel("$Q/a$",fontsize=fontsize)
        plot.set_ylabel("$1-e$",fontsize=fontsize)

        w=2.0
                    
        plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)

        loc = "lower left"
        handles,labels = plot.get_legend_handles_labels()
        plot.legend(handles,labels,loc=loc,fontsize=0.6*fontsize)

        plot.axhline(y=1.0e-2,color='k',linestyle='dotted',linewidth=w,zorder=0)
        plot.set_title("$E = %s; \, m_1=m_2=M; \,i=%s\,\mathrm{deg};\, \omega=%s\,\mathrm{deg};\, \Omega=%s\,\mathrm{deg}$"%(e_per,round(args.i*180.0/np.pi,1),round(args.omega*180.0/np.pi,1),round(args.Omega*180.0/np.pi,1)),fontsize=0.65*fontsize)
        
        fig.subplots_adjust(hspace=0.0,wspace=0.0)
        filename = os.path.dirname(os.path.realpath(__file__)) + '/figs/overview_' + str(args.name) + '_e_per_' + str(e_per) + '_m1_' + str(m1) + '_m2_' + str(m2) + '_M_per_' + str(M_per) + '_i_' + str(i) + '.pdf'
        fig.savefig(filename,dpi=200)


        
if __name__ == '__main__':
    args = parse_arguments()

#    if args.verbose==True:
#        print 'arguments:'
#        from pprint import pprint
#        pprint(vars(args))

    if args.plot>0:
        if HAS_MATPLOTLIB == False:
            print( 'Error importing Matplotlib -- choose --plot 0')
            exit(-1)

        if args.plot_fancy == True:
            pyplot.rc('text',usetex=True)
            pyplot.rc('legend',fancybox=True)


    if args.calc == True:
        if args.mode in [1,2]:
            data = core.integrate(args)
        elif args.mode in [3,4]:
            data_series = integrate_series(args)
        elif args.mode in [5]:
            data_series_angles = integrate_series_angles(args)
        elif args.mode in [6]:
            data_series_PN = integrate_series_PN(args)

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
            print( 'filename',filename)
            with open(filename, 'rb') as handle:
                data_series = pickle.load(handle)
            
            if args.mode==3:
                plot_function_series(args,data_series)
            if args.mode==4:
                plot_function_series_detailed(args,data_series)

        elif args.mode in [5]:
            filename = args.series_angles_data_filename
            print( 'filename',filename)
            with open(filename, 'rb') as handle:
                data_series_angles = pickle.load(handle)
            
            plot_function_series_angles(args,data_series_angles)

        elif args.mode in [6]:
            filename = args.series_PN_data_filename
            print( 'filename',filename)
            with open(filename, 'rb') as handle:
                data_series_PN = pickle.load(handle)
            
            plot_function_series_PN(args,data_series_PN)
        
        elif args.mode in [7]:
            plot_function_overview(args)
            

        else:
            'Incorrect mode'
            exit(-1)


    if args.plot>0 and args.show == True:
        pyplot.show()
