import ctypes

import os
path = os.path.dirname(os.path.realpath(__file__))
flybyslibrary = ctypes.CDLL(path+'/flybyslibrary.so')


"""
def triple_EOM_(CONST_G,CONST_C,e_in,e_out,g_in,g_out,cositot, \
    a_in,a_out,m1,m2,m3, \
    include_quadrupole_terms,include_octupole_terms,include_1PN_terms):

    de_in_dt = ctypes.c_double(0.0)
    de_out_dt = ctypes.c_double(0.0)
    dg_in_dt = ctypes.c_double(0.0)
    dg_out_dt = ctypes.c_double(0.0)
    dcositot_dt = ctypes.c_double(0.0)
    
    emtlibrary.triple_EOM_.argtypes = (ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,\
        ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,\
        ctypes.c_int,ctypes.c_int,ctypes.c_int,\
        ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),\
        ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))
    
    emtlibrary.triple_EOM_.restype = ctypes.c_void_p
    emtlibrary.triple_EOM_(CONST_G,CONST_C,e_in,e_out,g_in,g_out,cositot, \
        a_in,a_out,m1,m2,m3, \
        include_quadrupole_terms,include_octupole_terms,include_1PN_terms,\
        ctypes.byref(de_in_dt), ctypes.byref(de_out_dt),\
        ctypes.byref(dg_in_dt), ctypes.byref(dg_out_dt),
        ctypes.byref(dcositot_dt))
    
    return (de_in_dt.value, de_out_dt.value, dg_in_dt.value, dg_out_dt.value, dcositot_dt.value)

"""
f = flybyslibrary.f_e_cdot_e_hat
f.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
f.restype = ctypes.c_double


f = flybyslibrary.g_e_I_cdot_e_hat
f.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
f.restype = ctypes.c_double


f = flybyslibrary.g_e_II_cdot_e_hat
f.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
f.restype = ctypes.c_double
