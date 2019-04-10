#include <stdio.h>
#include <math.h>



/* Functions that appear in the right-hand-sides of the
 * orbit-averaged equations of motion for the `emt' model. 
 * Interfaced to Python using ctypes (uses wrapperemtlibrary.py)
 * Adrian Hamers
 * November 2018 */



double f_func(double n, double e_per);
double f_e_cdot_e_hat(double eps_SA, double e_per, double ex, double ey, double ez, double jx, double jy, double jz);
double g_e_I_cdot_e_hat(double eps_SA, double e_per, double ex, double ey, double ez, double jx, double jy, double jz);
double g_e_II_cdot_e_hat(double eps_SA, double e_per, double ex, double ey, double ez, double jx, double jy, double jz);


double f_func(double n, double e_per)
{
    double L = acos(-1.0/e_per);
    double npi = n*M_PI;
    return (npi - 3.0*L)*(npi - 2.0*L)*(npi - L)*(npi + L)*(npi + 2.0*L)*(npi + 3.0*L);
}

double f_e_cdot_e_hat(double eps_SA, double e_per, double ex, double ey, double ez, double jx, double jy, double jz)
{
    double L = acos(-1.0/e_per);
    double pow_e_per_2 = e_per*e_per;
    double sqrt_om_e_per_pm2 = sqrt(1.0 - 1.0/pow_e_per_2);
    
    return (5*(sqrt_om_e_per_pm2*((1 + 2*pow_e_per_2)*ey*ez*jx + (1 - 4*pow_e_per_2)*ex*ez*jy + 2*(-1 + pow_e_per_2)*ex*ey*jz) + 3*e_per*ez*(ey*jx - ex*jy)*L)*eps_SA)/(2.*e_per*sqrt(ex*ex + ey*ey + ez*ez));
}

double g_e_I_cdot_e_hat(double eps_SA, double e_per, double ex, double ey, double ez, double jx, double jy, double jz)
{
    double L = acos(-1.0/e_per);
    double pow_L_2 = L*L;
    double pow_L_3 = L*pow_L_2;
    double pow_e_per_2 = e_per*e_per;
    double sqrt_om_e_per_pm2 = sqrt(1.0 - 1.0/pow_e_per_2);

    double f1 = f_func(1.0,e_per);
    double f2 = f_func(2.0,e_per);
    double f3 = f_func(3.0,e_per);
    
    double pow_M_PI_2 = M_PI*M_PI;
    
    double pow_ex_2 = ex*ex;
    double pow_ey_2 = ey*ey;
    double pow_ez_2 = ez*ez;
    double pow_jx_2 = jx*jx;
    double pow_jy_2 = jy*jy;
    double pow_jz_2 = jz*jz;

    double exn = ex + eps_SA*((sqrt_om_e_per_pm2*(ez*(jy - 10*pow_e_per_2*jy) + (-5 + 2*pow_e_per_2)*ey*jz) - 3*e_per*(3*ez*jy + ey*jz)*L)/(4.*e_per) - (3*sqrt_om_e_per_pm2*(ez*jx - 5*ex*jz)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*pow_e_per_2)*pow_M_PI_2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3));
    double eyn = ey + eps_SA*((sqrt_om_e_per_pm2*(ez*(jx + 8*pow_e_per_2*jx) + (-5 + 8*pow_e_per_2)*ex*jz) + 3*e_per*(3*ez*jx + ex*jz)*L)/(4.*e_per) + (3*sqrt_om_e_per_pm2*(ez*jy - 5*ey*jz)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*pow_e_per_2)*pow_M_PI_2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3));
    double ezn = ez + eps_SA*(((3*(ey*jx - ex*jy))/2. + (sqrt_om_e_per_pm2*((2 + pow_e_per_2)*ey*jx + (2 - 5*pow_e_per_2)*ex*jy))/(2.*e_per*L))*L - (12*sqrt_om_e_per_pm2*(ex*jx - ey*jy)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*pow_e_per_2)*pow_M_PI_2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3));
    double jxn = jx + eps_SA*(-((5*ey*ez - jy*jz)*(sqrt_om_e_per_pm2*(1 + 2*pow_e_per_2) + 3*e_per*L))/(4.*e_per) + (3*sqrt_om_e_per_pm2*(5*ex*ez - jx*jz)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*pow_e_per_2)*pow_M_PI_2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3));
    double jyn = jy + eps_SA*(((5*ex*ez - jx*jz)*(sqrt_om_e_per_pm2*(-1 + 4*pow_e_per_2) + 3*e_per*L))/(4.*e_per) - (3*sqrt_om_e_per_pm2*(5*ey*ez - jy*jz)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*pow_e_per_2)*pow_M_PI_2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3));
    double jzn = jz + eps_SA*(-(sqrt_om_e_per_pm2*(-1 + pow_e_per_2)*(5*ex*ey - jx*jy))/(2.*e_per) - (3*sqrt_om_e_per_pm2*(5*pow_ex_2 - 5*pow_ey_2 - pow_jx_2 + pow_jy_2)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2*(f2*f3 + f1*(f2 + f3)) - (-5 + 2*pow_e_per_2)*pow_M_PI_2*(f2*f3 + f1*(9*f2 + 4*f3))))/(e_per*f1*f2*f3));

    return f_e_cdot_e_hat(eps_SA,e_per,exn,eyn,ezn,jxn,jyn,jzn);
}

double g_e_II_cdot_e_hat(double eps_SA, double e_per, double ex, double ey, double ez, double jx, double jy, double jz)
{

    double L = acos(-1.0/e_per);
    double pow_L_2 = L*L;
    double pow_L_3 = L*pow_L_2;
    double pow_L_4 = L*pow_L_3;
    double pow_L_5 = L*pow_L_4;
    double pow_e_per_2 = e_per*e_per;
    double sqrt_om_e_per_pm2 = sqrt(1.0 - 1.0/pow_e_per_2);
    
    double f1 = f_func(1.0,e_per);
    double f2 = f_func(2.0,e_per);
    double f3 = f_func(3.0,e_per);
    
    double pow_M_PI_2 = M_PI*M_PI;
    
    double pow_ex_2 = ex*ex;
    double pow_ey_2 = ey*ey;
    double pow_ez_2 = ez*ez;
    double pow_jx_2 = jx*jx;
    double pow_jy_2 = jy*jy;
    double pow_jz_2 = jz*jz;
    
/*    if (l_max==3)
    {
    }
*/
    /* Expression below assumes l_max = 3 */
    
    return (3*pow_L_2*pow(eps_SA,2)*((ey*(-18*(-1 + pow_e_per_2)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2*M_PI + 9*(5 - 2*pow_e_per_2)*pow(M_PI,3))*(2*(-2*jx*((-7 + 4*pow_e_per_2)*ey*jy + 2*(1 + 11*pow_e_per_2)*ez*jz) + ex*(50*(-1 + pow_e_per_2)*pow_ey_2 + 5*(-1 + 4*pow_e_per_2)*pow_ez_2 + 4*pow_jy_2 - 10*pow_e_per_2*pow_jy_2 + 25*pow_jz_2 - 40*pow_e_per_2*pow_jz_2))*pow_L_4 + 3*(2*jx*(-5*(7 + 2*pow_e_per_2)*ey*jy + (10 + 53*pow_e_per_2)*ez*jz) + ex*(25*(10 + pow_e_per_2)*pow_ey_2 - 5*(-5 + 6*pow_e_per_2)*pow_ez_2 - 20*pow_jy_2 + 11*pow_e_per_2*pow_jy_2 - 125*pow_jz_2 + 20*pow_e_per_2*pow_jz_2))*pow_L_2*pow_M_PI_2 - 27*(2*jx*((7 - 4*pow_e_per_2)*ey*jy + (-2 + 5*pow_e_per_2)*ez*jz) + ex*(25*(-2 + pow_e_per_2)*pow_ey_2 - 5*pow_ez_2 + 4*pow_jy_2 - pow_e_per_2*pow_jy_2 + 25*pow_jz_2 - 10*pow_e_per_2*pow_jz_2))*pow(M_PI,4))*pow(f1,2)*pow(f2,2) - 18*(-1 + pow_e_per_2)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2 + 9*(5 - 2*pow_e_per_2)*pow_M_PI_2)*(2*(5*(-5 + 8*pow_e_per_2)*pow(ex,3) - 4*jx*(ey*(jy + 8*pow_e_per_2*jy) + (1 - 4*pow_e_per_2)*ez*jz) + ex*(-5*(-5 + 8*pow_e_per_2)*pow_ey_2 - 5*(1 + 8*pow_e_per_2)*pow_ez_2 + 9*pow_jx_2 + 24*pow_e_per_2*pow_jx_2 - 5*pow_jy_2 + 8*pow_e_per_2*pow_jy_2 + 25*pow_jz_2 - 40*pow_e_per_2*pow_jz_2))*pow_L_4*M_PI - 3*(5*(-25 + 4*pow_e_per_2)*pow(ex,3) - 4*jx*(5*ey*(jy + 4*pow_e_per_2*jy) + (5 - 6*pow_e_per_2)*ez*jz) + ex*(-5*(-25 + 4*pow_e_per_2)*pow_ey_2 - 25*(1 + 4*pow_e_per_2)*pow_ez_2 + 45*pow_jx_2 + 76*pow_e_per_2*pow_jx_2 - 25*pow_jy_2 + 4*pow_e_per_2*pow_jy_2 + 125*pow_jz_2 - 20*pow_e_per_2*pow_jz_2))*pow_L_2*pow(M_PI,3) - 27*(5*(-5 + 2*pow_e_per_2)*pow(ex,3) + 4*jx*((-1 + 2*pow_e_per_2)*ey*jy - ez*jz) + ex*(-5*(-5 + 2*pow_e_per_2)*pow_ey_2 + 5*(-1 + 2*pow_e_per_2)*pow_ez_2 + 9*pow_jx_2 - 10*pow_e_per_2*pow_jx_2 - 5*pow_jy_2 + 2*pow_e_per_2*pow_jy_2 + 25*pow_jz_2 - 10*pow_e_per_2*pow_jz_2))*pow(M_PI,5))*pow(f1,2)*pow(f2,2) + sqrt_om_e_per_pm2*pow_e_per_2*L*(sqrt_om_e_per_pm2*(-2*jx*((-7 + 4*pow_e_per_2)*ey*jy + 2*(1 + 11*pow_e_per_2)*ez*jz) + ex*(50*(-1 + pow_e_per_2)*pow_ey_2 + 5*(-1 + 4*pow_e_per_2)*pow_ez_2 + 4*pow_jy_2 - 10*pow_e_per_2*pow_jy_2 + 25*pow_jz_2 - 40*pow_e_per_2*pow_jz_2)) - 15*e_per*jz*(3*ez*jx + ex*jz)*L + 3*e_per*(5*ex*pow_ez_2 + 2*ey*jx*jy - 2*ex*pow_jy_2 - ez*jx*jz)*L)*M_PI*((5 - 8*pow_e_per_2)*pow_L_2 + 9*(-5 + 2*pow_e_per_2)*pow_M_PI_2)*pow(f1,2)*pow(f2,2)*f3 - 12*(-1 + pow_e_per_2)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2*M_PI + 4*(5 - 2*pow_e_per_2)*pow(M_PI,3))*(3*(-2*jx*((-7 + 4*pow_e_per_2)*ey*jy + 2*(1 + 11*pow_e_per_2)*ez*jz) + ex*(50*(-1 + pow_e_per_2)*pow_ey_2 + 5*(-1 + 4*pow_e_per_2)*pow_ez_2 + 4*pow_jy_2 - 10*pow_e_per_2*pow_jy_2 + 25*pow_jz_2 - 40*pow_e_per_2*pow_jz_2))*pow_L_4 + 2*(2*jx*(-5*(7 + 2*pow_e_per_2)*ey*jy + (10 + 53*pow_e_per_2)*ez*jz) + ex*(25*(10 + pow_e_per_2)*pow_ey_2 - 5*(-5 + 6*pow_e_per_2)*pow_ez_2 - 20*pow_jy_2 + 11*pow_e_per_2*pow_jy_2 - 125*pow_jz_2 + 20*pow_e_per_2*pow_jz_2))*pow_L_2*pow_M_PI_2 - 8*(2*jx*((7 - 4*pow_e_per_2)*ey*jy + (-2 + 5*pow_e_per_2)*ez*jz) + ex*(25*(-2 + pow_e_per_2)*pow_ey_2 - 5*pow_ez_2 + 4*pow_jy_2 - pow_e_per_2*pow_jy_2 + 25*pow_jz_2 - 10*pow_e_per_2*pow_jz_2))*pow(M_PI,4))*pow(f1,2)*pow(f3,2) - 12*(-1 + pow_e_per_2)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2 + 4*(5 - 2*pow_e_per_2)*pow_M_PI_2)*(3*(5*(-5 + 8*pow_e_per_2)*pow(ex,3) - 4*jx*(ey*(jy + 8*pow_e_per_2*jy) + (1 - 4*pow_e_per_2)*ez*jz) + ex*(-5*(-5 + 8*pow_e_per_2)*pow_ey_2 - 5*(1 + 8*pow_e_per_2)*pow_ez_2 + 9*pow_jx_2 + 24*pow_e_per_2*pow_jx_2 - 5*pow_jy_2 + 8*pow_e_per_2*pow_jy_2 + 25*pow_jz_2 - 40*pow_e_per_2*pow_jz_2))*pow_L_4*M_PI + 2*(-5*(-25 + 4*pow_e_per_2)*pow(ex,3) + 4*jx*(5*ey*(jy + 4*pow_e_per_2*jy) + (5 - 6*pow_e_per_2)*ez*jz) + ex*(5*(-25 + 4*pow_e_per_2)*pow_ey_2 + 25*(1 + 4*pow_e_per_2)*pow_ez_2 - 45*pow_jx_2 - 76*pow_e_per_2*pow_jx_2 + 25*pow_jy_2 - 4*pow_e_per_2*pow_jy_2 - 125*pow_jz_2 + 20*pow_e_per_2*pow_jz_2))*pow_L_2*pow(M_PI,3) - 8*(5*(-5 + 2*pow_e_per_2)*pow(ex,3) + 4*jx*((-1 + 2*pow_e_per_2)*ey*jy - ez*jz) + ex*(-5*(-5 + 2*pow_e_per_2)*pow_ey_2 + 5*(-1 + 2*pow_e_per_2)*pow_ez_2 + 9*pow_jx_2 - 10*pow_e_per_2*pow_jx_2 - 5*pow_jy_2 + 2*pow_e_per_2*pow_jy_2 + 25*pow_jz_2 - 10*pow_e_per_2*pow_jz_2))*pow(M_PI,5))*pow(f1,2)*pow(f3,2) + sqrt_om_e_per_pm2*pow_e_per_2*L*(sqrt_om_e_per_pm2*(-2*jx*((-7 + 4*pow_e_per_2)*ey*jy + 2*(1 + 11*pow_e_per_2)*ez*jz) + ex*(50*(-1 + pow_e_per_2)*pow_ey_2 + 5*(-1 + 4*pow_e_per_2)*pow_ez_2 + 4*pow_jy_2 - 10*pow_e_per_2*pow_jy_2 + 25*pow_jz_2 - 40*pow_e_per_2*pow_jz_2)) - 15*e_per*jz*(3*ez*jx + ex*jz)*L + 3*e_per*(5*ex*pow_ez_2 + 2*ey*jx*jy - 2*ex*pow_jy_2 - ez*jx*jz)*L)*M_PI*((5 - 8*pow_e_per_2)*pow_L_2 + 4*(-5 + 2*pow_e_per_2)*pow_M_PI_2)*pow(f1,2)*f2*pow(f3,2) - 6*(-1 + pow_e_per_2)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2*M_PI + (5 - 2*pow_e_per_2)*pow(M_PI,3))*(6*(-2*jx*((-7 + 4*pow_e_per_2)*ey*jy + 2*(1 + 11*pow_e_per_2)*ez*jz) + ex*(50*(-1 + pow_e_per_2)*pow_ey_2 + 5*(-1 + 4*pow_e_per_2)*pow_ez_2 + 4*pow_jy_2 - 10*pow_e_per_2*pow_jy_2 + 25*pow_jz_2 - 40*pow_e_per_2*pow_jz_2))*pow_L_4 + (2*jx*(-5*(7 + 2*pow_e_per_2)*ey*jy + (10 + 53*pow_e_per_2)*ez*jz) + ex*(25*(10 + pow_e_per_2)*pow_ey_2 - 5*(-5 + 6*pow_e_per_2)*pow_ez_2 - 20*pow_jy_2 + 11*pow_e_per_2*pow_jy_2 - 125*pow_jz_2 + 20*pow_e_per_2*pow_jz_2))*pow_L_2*pow_M_PI_2 + (2*jx*((-7 + 4*pow_e_per_2)*ey*jy + (2 - 5*pow_e_per_2)*ez*jz) + ex*(-25*(-2 + pow_e_per_2)*pow_ey_2 + 5*pow_ez_2 - 4*pow_jy_2 + pow_e_per_2*pow_jy_2 - 25*pow_jz_2 + 10*pow_e_per_2*pow_jz_2))*pow(M_PI,4))*pow(f2,2)*pow(f3,2) - 6*(-1 + pow_e_per_2)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2 + (5 - 2*pow_e_per_2)*pow_M_PI_2)*(6*(5*(-5 + 8*pow_e_per_2)*pow(ex,3) - 4*jx*(ey*(jy + 8*pow_e_per_2*jy) + (1 - 4*pow_e_per_2)*ez*jz) + ex*(-5*(-5 + 8*pow_e_per_2)*pow_ey_2 - 5*(1 + 8*pow_e_per_2)*pow_ez_2 + 9*pow_jx_2 + 24*pow_e_per_2*pow_jx_2 - 5*pow_jy_2 + 8*pow_e_per_2*pow_jy_2 + 25*pow_jz_2 - 40*pow_e_per_2*pow_jz_2))*pow_L_4*M_PI + (-5*(-25 + 4*pow_e_per_2)*pow(ex,3) + 4*jx*(5*ey*(jy + 4*pow_e_per_2*jy) + (5 - 6*pow_e_per_2)*ez*jz) + ex*(5*(-25 + 4*pow_e_per_2)*pow_ey_2 + 25*(1 + 4*pow_e_per_2)*pow_ez_2 - 45*pow_jx_2 - 76*pow_e_per_2*pow_jx_2 + 25*pow_jy_2 - 4*pow_e_per_2*pow_jy_2 - 125*pow_jz_2 + 20*pow_e_per_2*pow_jz_2))*pow_L_2*pow(M_PI,3) + (-5*(-5 + 2*pow_e_per_2)*pow(ex,3) + 4*jx*(ey*(jy - 2*pow_e_per_2*jy) + ez*jz) + ex*(5*(-5 + 2*pow_e_per_2)*pow_ey_2 + (5 - 10*pow_e_per_2)*pow_ez_2 - 9*pow_jx_2 + 10*pow_e_per_2*pow_jx_2 + 5*pow_jy_2 - 2*pow_e_per_2*pow_jy_2 - 25*pow_jz_2 + 10*pow_e_per_2*pow_jz_2))*pow(M_PI,5))*pow(f2,2)*pow(f3,2) + sqrt_om_e_per_pm2*pow_e_per_2*L*(sqrt_om_e_per_pm2*(-2*jx*((-7 + 4*pow_e_per_2)*ey*jy + 2*(1 + 11*pow_e_per_2)*ez*jz) + ex*(50*(-1 + pow_e_per_2)*pow_ey_2 + 5*(-1 + 4*pow_e_per_2)*pow_ez_2 + 4*pow_jy_2 - 10*pow_e_per_2*pow_jy_2 + 25*pow_jz_2 - 40*pow_e_per_2*pow_jz_2)) - 15*e_per*jz*(3*ez*jx + ex*jz)*L + 3*e_per*(5*ex*pow_ez_2 + 2*ey*jx*jy - 2*ex*pow_jy_2 - ez*jx*jz)*L)*M_PI*((5 - 8*pow_e_per_2)*pow_L_2 + (-5 + 2*pow_e_per_2)*pow_M_PI_2)*f1*pow(f2,2)*pow(f3,2)))/M_PI + (ex*(18*(-1 + pow_e_per_2)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2*M_PI + 9*(5 - 2*pow_e_per_2)*pow(M_PI,3))*(2*(50*(-1 + pow_e_per_2)*pow_ex_2*ey + 2*(7 - 10*pow_e_per_2)*ex*jx*jy + 4*(-1 + 13*pow_e_per_2)*ez*jy*jz + ey*(-5*(1 + 2*pow_e_per_2)*pow_ez_2 + 2*(2 + pow_e_per_2)*pow_jx_2 + 5*(5 - 2*pow_e_per_2)*pow_jz_2))*pow_L_4 + 3*(25*(10 + pow_e_per_2)*pow_ex_2*ey + 2*(-35 + 3*pow_e_per_2)*ex*jx*jy + 2*(10 - 51*pow_e_per_2)*ez*jy*jz + 5*ey*((5 + 7*pow_e_per_2)*pow_ez_2 - (4 + 3*pow_e_per_2)*pow_jx_2 - (25 + 9*pow_e_per_2)*pow_jz_2))*pow_L_2*pow_M_PI_2 - 27*(25*(-2 + pow_e_per_2)*pow_ex_2*ey + 2*(7 - 3*pow_e_per_2)*ex*jx*jy - 2*(2 + 3*pow_e_per_2)*ez*jy*jz + ey*(5*(-1 + pow_e_per_2)*pow_ez_2 + (4 - 3*pow_e_per_2)*pow_jx_2 + 5*(5 - 3*pow_e_per_2)*pow_jz_2))*pow(M_PI,4))*pow(f1,2)*pow(f2,2) - 18*(-1 + pow_e_per_2)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2 + 9*(5 - 2*pow_e_per_2)*pow_M_PI_2)*(2*(5*(-5 + 2*pow_e_per_2)*pow_ex_2*ey - 5*(-5 + 2*pow_e_per_2)*pow(ey,3) + 4*(1 - 10*pow_e_per_2)*ex*jx*jy + 4*(1 + 2*pow_e_per_2)*ez*jy*jz + ey*((5 - 50*pow_e_per_2)*pow_ez_2 + (5 - 2*pow_e_per_2)*pow_jx_2 - 9*pow_jy_2 + 42*pow_e_per_2*pow_jy_2 - 25*pow_jz_2 + 10*pow_e_per_2*pow_jz_2))*pow_L_4*M_PI + 3*(5*(25 + 9*pow_e_per_2)*pow_ex_2*ey - 5*(25 + 9*pow_e_per_2)*pow(ey,3) + 4*(-5 + 19*pow_e_per_2)*ex*jx*jy - 4*(5 + 7*pow_e_per_2)*ez*jy*jz + ey*(5*(-5 + 19*pow_e_per_2)*pow_ez_2 - (25 + 9*pow_e_per_2)*pow_jx_2 + 45*pow_jy_2 - 67*pow_e_per_2*pow_jy_2 + 125*pow_jz_2 + 45*pow_e_per_2*pow_jz_2))*pow_L_2*pow(M_PI,3) - 27*(5*(-5 + 3*pow_e_per_2)*pow_ex_2*ey - 5*(-5 + 3*pow_e_per_2)*pow(ey,3) + 4*(1 + pow_e_per_2)*ex*jx*jy - 4*(-1 + pow_e_per_2)*ez*jy*jz + ey*(5*(1 + pow_e_per_2)*pow_ez_2 + (5 - 3*pow_e_per_2)*pow_jx_2 - 9*pow_jy_2 - pow_e_per_2*pow_jy_2 - 25*pow_jz_2 + 15*pow_e_per_2*pow_jz_2))*pow(M_PI,5))*pow(f1,2)*pow(f2,2) + sqrt_om_e_per_pm2*pow_e_per_2*L*(sqrt_om_e_per_pm2*(-50*(-1 + pow_e_per_2)*pow_ex_2*ey + 2*(-7 + 10*pow_e_per_2)*ex*jx*jy + 4*(1 - 13*pow_e_per_2)*ez*jy*jz + ey*(5*(1 + 2*pow_e_per_2)*pow_ez_2 - 2*(2 + pow_e_per_2)*pow_jx_2 + 5*(-5 + 2*pow_e_per_2)*pow_jz_2)) - 15*e_per*jz*(3*ez*jy + ey*jz)*L + 3*e_per*(5*ey*pow_ez_2 - 2*ey*pow_jx_2 + 2*ex*jx*jy - ez*jy*jz)*L)*M_PI*((5 - 8*pow_e_per_2)*pow_L_2 + 9*(-5 + 2*pow_e_per_2)*pow_M_PI_2)*pow(f1,2)*pow(f2,2)*f3 + 12*(-1 + pow_e_per_2)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2*M_PI + 4*(5 - 2*pow_e_per_2)*pow(M_PI,3))*(3*(50*(-1 + pow_e_per_2)*pow_ex_2*ey + 2*(7 - 10*pow_e_per_2)*ex*jx*jy + 4*(-1 + 13*pow_e_per_2)*ez*jy*jz + ey*(-5*(1 + 2*pow_e_per_2)*pow_ez_2 + 2*(2 + pow_e_per_2)*pow_jx_2 + 5*(5 - 2*pow_e_per_2)*pow_jz_2))*pow_L_4 + 2*(25*(10 + pow_e_per_2)*pow_ex_2*ey + 2*(-35 + 3*pow_e_per_2)*ex*jx*jy + 2*(10 - 51*pow_e_per_2)*ez*jy*jz + 5*ey*((5 + 7*pow_e_per_2)*pow_ez_2 - (4 + 3*pow_e_per_2)*pow_jx_2 - (25 + 9*pow_e_per_2)*pow_jz_2))*pow_L_2*pow_M_PI_2 - 8*(25*(-2 + pow_e_per_2)*pow_ex_2*ey + 2*(7 - 3*pow_e_per_2)*ex*jx*jy - 2*(2 + 3*pow_e_per_2)*ez*jy*jz + ey*(5*(-1 + pow_e_per_2)*pow_ez_2 + (4 - 3*pow_e_per_2)*pow_jx_2 + 5*(5 - 3*pow_e_per_2)*pow_jz_2))*pow(M_PI,4))*pow(f1,2)*pow(f3,2) - 12*(-1 + pow_e_per_2)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2 + 4*(5 - 2*pow_e_per_2)*pow_M_PI_2)*(3*(5*(-5 + 2*pow_e_per_2)*pow_ex_2*ey - 5*(-5 + 2*pow_e_per_2)*pow(ey,3) + 4*(1 - 10*pow_e_per_2)*ex*jx*jy + 4*(1 + 2*pow_e_per_2)*ez*jy*jz + ey*((5 - 50*pow_e_per_2)*pow_ez_2 + (5 - 2*pow_e_per_2)*pow_jx_2 - 9*pow_jy_2 + 42*pow_e_per_2*pow_jy_2 - 25*pow_jz_2 + 10*pow_e_per_2*pow_jz_2))*pow_L_4*M_PI + 2*(5*(25 + 9*pow_e_per_2)*pow_ex_2*ey - 5*(25 + 9*pow_e_per_2)*pow(ey,3) + 4*(-5 + 19*pow_e_per_2)*ex*jx*jy - 4*(5 + 7*pow_e_per_2)*ez*jy*jz + ey*(5*(-5 + 19*pow_e_per_2)*pow_ez_2 - (25 + 9*pow_e_per_2)*pow_jx_2 + 45*pow_jy_2 - 67*pow_e_per_2*pow_jy_2 + 125*pow_jz_2 + 45*pow_e_per_2*pow_jz_2))*pow_L_2*pow(M_PI,3) - 8*(5*(-5 + 3*pow_e_per_2)*pow_ex_2*ey - 5*(-5 + 3*pow_e_per_2)*pow(ey,3) + 4*(1 + pow_e_per_2)*ex*jx*jy - 4*(-1 + pow_e_per_2)*ez*jy*jz + ey*(5*(1 + pow_e_per_2)*pow_ez_2 + (5 - 3*pow_e_per_2)*pow_jx_2 - 9*pow_jy_2 - pow_e_per_2*pow_jy_2 - 25*pow_jz_2 + 15*pow_e_per_2*pow_jz_2))*pow(M_PI,5))*pow(f1,2)*pow(f3,2) + sqrt_om_e_per_pm2*pow_e_per_2*L*(sqrt_om_e_per_pm2*(-50*(-1 + pow_e_per_2)*pow_ex_2*ey + 2*(-7 + 10*pow_e_per_2)*ex*jx*jy + 4*(1 - 13*pow_e_per_2)*ez*jy*jz + ey*(5*(1 + 2*pow_e_per_2)*pow_ez_2 - 2*(2 + pow_e_per_2)*pow_jx_2 + 5*(-5 + 2*pow_e_per_2)*pow_jz_2)) - 15*e_per*jz*(3*ez*jy + ey*jz)*L + 3*e_per*(5*ey*pow_ez_2 - 2*ey*pow_jx_2 + 2*ex*jx*jy - ez*jy*jz)*L)*M_PI*((5 - 8*pow_e_per_2)*pow_L_2 + 4*(-5 + 2*pow_e_per_2)*pow_M_PI_2)*pow(f1,2)*f2*pow(f3,2) + 6*(-1 + pow_e_per_2)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2*M_PI + (5 - 2*pow_e_per_2)*pow(M_PI,3))*(6*(50*(-1 + pow_e_per_2)*pow_ex_2*ey + 2*(7 - 10*pow_e_per_2)*ex*jx*jy + 4*(-1 + 13*pow_e_per_2)*ez*jy*jz + ey*(-5*(1 + 2*pow_e_per_2)*pow_ez_2 + 2*(2 + pow_e_per_2)*pow_jx_2 + 5*(5 - 2*pow_e_per_2)*pow_jz_2))*pow_L_4 + (25*(10 + pow_e_per_2)*pow_ex_2*ey + 2*(-35 + 3*pow_e_per_2)*ex*jx*jy + 2*(10 - 51*pow_e_per_2)*ez*jy*jz + 5*ey*((5 + 7*pow_e_per_2)*pow_ez_2 - (4 + 3*pow_e_per_2)*pow_jx_2 - (25 + 9*pow_e_per_2)*pow_jz_2))*pow_L_2*pow_M_PI_2 + (-25*(-2 + pow_e_per_2)*pow_ex_2*ey + 2*(-7 + 3*pow_e_per_2)*ex*jx*jy + 2*(2 + 3*pow_e_per_2)*ez*jy*jz + ey*(-5*(-1 + pow_e_per_2)*pow_ez_2 + (-4 + 3*pow_e_per_2)*pow_jx_2 + 5*(-5 + 3*pow_e_per_2)*pow_jz_2))*pow(M_PI,4))*pow(f2,2)*pow(f3,2) - 6*(-1 + pow_e_per_2)*pow_L_3*((-5 + 8*pow_e_per_2)*pow_L_2 + (5 - 2*pow_e_per_2)*pow_M_PI_2)*(6*(5*(-5 + 2*pow_e_per_2)*pow_ex_2*ey - 5*(-5 + 2*pow_e_per_2)*pow(ey,3) + 4*(1 - 10*pow_e_per_2)*ex*jx*jy + 4*(1 + 2*pow_e_per_2)*ez*jy*jz + ey*((5 - 50*pow_e_per_2)*pow_ez_2 + (5 - 2*pow_e_per_2)*pow_jx_2 - 9*pow_jy_2 + 42*pow_e_per_2*pow_jy_2 - 25*pow_jz_2 + 10*pow_e_per_2*pow_jz_2))*pow_L_4*M_PI + (5*(25 + 9*pow_e_per_2)*pow_ex_2*ey - 5*(25 + 9*pow_e_per_2)*pow(ey,3) + 4*(-5 + 19*pow_e_per_2)*ex*jx*jy - 4*(5 + 7*pow_e_per_2)*ez*jy*jz + ey*(5*(-5 + 19*pow_e_per_2)*pow_ez_2 - (25 + 9*pow_e_per_2)*pow_jx_2 + 45*pow_jy_2 - 67*pow_e_per_2*pow_jy_2 + 125*pow_jz_2 + 45*pow_e_per_2*pow_jz_2))*pow_L_2*pow(M_PI,3) + (-5*(-5 + 3*pow_e_per_2)*pow_ex_2*ey + 5*(-5 + 3*pow_e_per_2)*pow(ey,3) - 4*(1 + pow_e_per_2)*ex*jx*jy + 4*(-1 + pow_e_per_2)*ez*jy*jz + ey*(-5*(1 + pow_e_per_2)*pow_ez_2 + (-5 + 3*pow_e_per_2)*pow_jx_2 + 9*pow_jy_2 + pow_e_per_2*pow_jy_2 + 25*pow_jz_2 - 15*pow_e_per_2*pow_jz_2))*pow(M_PI,5))*pow(f2,2)*pow(f3,2) + sqrt_om_e_per_pm2*pow_e_per_2*L*(sqrt_om_e_per_pm2*(-50*(-1 + pow_e_per_2)*pow_ex_2*ey + 2*(-7 + 10*pow_e_per_2)*ex*jx*jy + 4*(1 - 13*pow_e_per_2)*ez*jy*jz + ey*(5*(1 + 2*pow_e_per_2)*pow_ez_2 - 2*(2 + pow_e_per_2)*pow_jx_2 + 5*(-5 + 2*pow_e_per_2)*pow_jz_2)) - 15*e_per*jz*(3*ez*jy + ey*jz)*L + 3*e_per*(5*ey*pow_ez_2 - 2*ey*pow_jx_2 + 2*ex*jx*jy - ez*jy*jz)*L)*M_PI*((5 - 8*pow_e_per_2)*pow_L_2 + (-5 + 2*pow_e_per_2)*pow_M_PI_2)*f1*pow(f2,2)*pow(f3,2)))/M_PI - 12*sqrt_om_e_per_pm2*pow_e_per_2*ez*((-5 + 8*pow_e_per_2)*pow_L_3*(2*sqrt_om_e_per_pm2*((ey*jx - ex*jy)*jz + pow_e_per_2*(5*ex*ey*ez + 3*ez*jx*jy - ey*jx*jz + ex*jy*jz)) + e_per*(6*ez*jx*jy + ey*jx*jz + ex*jy*jz)*L)*f1*f2*f3*(f2*f3 + f1*(f2 + f3)) - e_per*(-5 + 2*pow_e_per_2)*(10*ex*ey*ez - ey*jx*jz - ex*jy*jz)*pow_L_2*pow_M_PI_2*f1*f2*f3*(f2*f3 + f1*(9*f2 + 4*f3)) - (-5 + 2*pow_e_per_2)*L*(2*sqrt_om_e_per_pm2*((ey*jx - ex*jy)*jz + pow_e_per_2*(5*ex*ey*ez + 3*ez*jx*jy - ey*jx*jz + ex*jy*jz)) + e_per*(6*ez*jx*jy + ey*jx*jz + ex*jy*jz)*L)*pow_M_PI_2*f1*f2*f3*(f2*f3 + f1*(9*f2 + 4*f3)) + 36*sqrt_om_e_per_pm2*(-5 + 8*pow_e_per_2)*(4*(ey*jx - ex*jy)*jz + pow_e_per_2*(5*ex*ey*ez + 5*ez*jx*jy - ey*jx*jz + 7*ex*jy*jz))*pow(L,9)*(pow(f2,2)*pow(f3,2) + pow(f1,2)*(pow(f2,2) + pow(f3,2))) - sqrt_om_e_per_pm2*(1320*(-(ey*jx*jz) + ex*jy*jz) + 16*pow(e_per,4)*(55*ex*ey*ez + 55*ez*jx*jy + 21*ey*jx*jz + 45*ex*jy*jz) - pow_e_per_2*(1225*ex*ey*ez + 1225*ez*jx*jy - 1173*ey*jx*jz + 2643*ex*jy*jz))*pow(L,7)*pow_M_PI_2*(pow(f2,2)*pow(f3,2) + pow(f1,2)*(9*pow(f2,2) + 4*pow(f3,2))) + 2*sqrt_om_e_per_pm2*(pow(e_per,4)*(85*ex*ey*ez + 85*ez*jx*jy + 111*ey*jx*jz - 9*ex*jy*jz) + 240*(-(ey*jx*jz) + ex*jy*jz) - pow_e_per_2*(175*ex*ey*ez + 175*ez*jx*jy + 141*ey*jx*jz + 69*ex*jy*jz))*pow_L_5*pow(M_PI,4)*(pow(f2,2)*pow(f3,2) + pow(f1,2)*(81*pow(f2,2) + 16*pow(f3,2))) + pow_L_3*(e_per*(-5 + 8*pow_e_per_2)*(10*ex*ey*ez - ey*jx*jz - ex*jy*jz)*L*f1*f2*f3*(f2*f3 + f1*(f2 + f3)) - sqrt_om_e_per_pm2*(-5 + 2*pow_e_per_2)*(24*(-(ey*jx) + ex*jy)*jz + pow_e_per_2*(5*ex*ey*ez + 5*ez*jx*jy + 15*ey*jx*jz - 9*ex*jy*jz))*pow(M_PI,6)*(pow(f2,2)*pow(f3,2) + pow(f1,2)*(729*pow(f2,2) + 64*pow(f3,2)))))))/(2.*pow(e_per,4)*sqrt(pow_ex_2 + pow_ey_2 + pow_ez_2)*pow(f1,2)*pow(f2,2)*pow(f3,2));
}