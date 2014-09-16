#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define MYDIV(P,Q)  ((P) < 0 ? ((P)+1)/(Q)-1 : (P)/(Q))
#define ABS(A) ((A)<0 ? (-1.0)*(A) : A)
#define ANGDIFF(A,B) (ABS(A-B)<ABS(1.0-ABS(A-B)) ? A-B : ( (A-B)>0.0 ? (A-B)-1.0 : (ABS(A-B)<0.5 ? B-A : 1.0+(A-B)) )  )

<MACROS>

// MWC64X by David B. Thomas
ulong MWC_AddMod64(ulong a, ulong b, ulong M) {
  ulong v=a+b;
  if( (v>=M) || (v<a) )
    v=v-M;
  return v;
}
ulong MWC_MulMod64(ulong a, ulong b, ulong M) {
  ulong r=0;
  while(a!=0){
    if(a&1)
      r=MWC_AddMod64(r,b,M);
    b=MWC_AddMod64(b,b,M);
    a=a>>1;
  }
  return r;
}
ulong MWC_PowMod64(ulong a, ulong e, ulong M) {
  ulong sqr=a, acc=1;
  while(e!=0){
    if(e&1)
      acc=MWC_MulMod64(acc,sqr,M);
    sqr=MWC_MulMod64(sqr,sqr,M);
    e=e>>1;
  }
  return acc;
}
uint2 MWC_SkipImpl_Mod64(uint2 curr, ulong A, ulong M, ulong distance) {
  ulong m=MWC_PowMod64(A, distance, M);
  ulong x=curr.x*(ulong)A+curr.y;
  x=MWC_MulMod64(x, m, M);
  return (uint2)((uint)(x/A), (uint)(x%A));
}
uint2 MWC_SeedImpl_Mod64(ulong A, ulong M, uint vecSize, uint vecOffset, ulong streamBase, ulong streamGap) {
  enum{ MWC_BASEID = 4077358422479273989UL };

  ulong dist=streamBase + (get_global_id(0)*vecSize+vecOffset)*streamGap;
  ulong m=MWC_PowMod64(A, dist, M);

  ulong x=MWC_MulMod64(MWC_BASEID, m, M);
  return (uint2)((uint)(x/A), (uint)(x%A));
}

typedef struct{ uint x; uint c; } mwc64x_state_t;

enum{ MWC64X_A = 4294883355U };
enum{ MWC64X_M = 18446383549859758079UL };

void MWC64X_Step(mwc64x_state_t *s) {
  uint X=s->x, C=s->c;

  uint Xn=MWC64X_A*X+C;
  uint carry=(uint)(Xn<C);
  uint Cn=mad_hi(MWC64X_A,X,carry);

  s->x=Xn;
  s->c=Cn;
}
void MWC64X_Skip(mwc64x_state_t *s, ulong distance) {
  uint2 tmp=MWC_SkipImpl_Mod64((uint2)(s->x,s->c), MWC64X_A, MWC64X_M, distance);
  s->x=tmp.x;
  s->c=tmp.y;
}
void MWC64X_SeedStreams(mwc64x_state_t *s, ulong baseOffset, ulong perStreamOffset) {
  uint2 tmp=MWC_SeedImpl_Mod64(MWC64X_A, MWC64X_M, 1, 0, baseOffset, perStreamOffset);
  s->x=tmp.x;
  s->c=tmp.y;
}
uint MWC64X_NextUint(mwc64x_state_t *s) {
  uint res=s->x ^ s->c;
  MWC64X_Step(s);
  return res;
}

__kernel void init_MWC64X(__global mwc64x_state_t* RandState, const uint Seed) {
  uint n = get_global_id(0);

  mwc64x_state_t rng;

  MWC64X_SeedStreams(&rng, Seed, LINSIZE*NUM_OF_STEPS*8);

  RandState[n] = rng;
}

tFloat random_normal(mwc64x_state_t* rng) {
  mwc64x_state_t state = *rng;

  float u1 = (float)MWC64X_NextUint(&state) / (float)UINT_MAX;
  float u2 = (float)MWC64X_NextUint(&state) / (float)UINT_MAX;

  float u = max(0.000001f, u1);
  float v = max(0.000001f, u2);

  *rng = state;

  return sqrt(-2 * log(u)) * cos(2 * M_PI_F * v);
}

tFloat random_uniform(mwc64x_state_t* s){
  uint res=s->x ^ s->c;
  MWC64X_Step(s);
  return (float)res/(float)UINT_MAX;
}



//************************************************
// Mersenne Twister PRNG
// #define   MT_RNG_COUNT 4096
// #define          MT_MM 9
// #define          MT_NN 19
// #define       MT_WMASK 0xFFFFFFFFU
// #define       MT_UMASK 0xFFFFFFFEU
// #define       MT_LMASK 0x1U
// #define      MT_SHIFT0 12
// #define      MT_SHIFTB 7
// #define      MT_SHIFTC 15
// #define      MT_SHIFT1 18
 #define PI 3.14159265358979f




// double random_uniform(int globalID, int* seed) {
//     int iState, iState1, iStateM, iOut;
//     unsigned int mti, mti1, mtiM, x;
//     unsigned int mt[MT_NN], matrix_a, mask_b, mask_c;
//     double ret;

//     matrix_a = -822663744+globalID;
//     mask_b   = -1514742400;
//     mask_c   = -2785280+globalID;

//     //Initialize current state
//     mt[0] = *seed;
//     for (iState = 1; iState < MT_NN; iState++)
//       mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;

//     iState = 0;
//     mti1 = mt[0];
//     // for (iOut = 0; iOut < nPerRng; iOut++) {
//       iState1 = iState + 1;
//       iStateM = iState + MT_MM;
//       if(iState1 >= MT_NN) iState1 -= MT_NN;
//       if(iStateM >= MT_NN) iStateM -= MT_NN;
//       mti  = mti1;
//       mti1 = mt[iState1];
//       mtiM = mt[iStateM];

//       // MT recurrence
//       x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
//       x = mtiM ^ (x >> 1) ^ ((x & 1) ? matrix_a : 0);

//       mt[iState] = x;
//       iState = iState1;

//       //Tempering transformation
//       x ^= (x >> MT_SHIFT0);
//       x ^= (x << MT_SHIFTB) & mask_b;
//       x ^= (x << MT_SHIFTC) & mask_c;
//       x ^= (x >> MT_SHIFT1);

//       *seed = x;

//       //Convert to (0, 1] float and write to global memory
//       ret = (double)(((float)x + 1.0f) / 4294967296.0f);
//     // }

//   return ret;
// }


// double random_normal(int globalID, int* seed, double sigma) {
//   int state = *seed;
//   double u1 = random_uniform(globalID, &state);
//   double u2 = random_uniform(globalID, &state);

//   *seed = state;

//   double   r = sqrt(-2.0 * log(u1));
//   double phi = 2 * PI * (u2);

//   return (sigma * r * cos(phi));
// }

// /**
//  * Init PRNG seed (Pseudo Random Number Generator)
//  */
// __kernel void init_seed(__global int* Seed) {
//   int PHI_NOISE_SEED=128;
//   int i, n = get_global_id(0);

//   int seed = (int) PHI_NOISE_SEED + n;

//   double const a    = 16807+(double)seed;      // 7**5
//   double const m    = 2147483647; // 2**31-1
//   double const reciprocal_m = 1.0/m;

//   double temp;

//   for(i=0; i<1000; i++) {
//     temp = seed * a;
//     seed = (int) (temp - m * floor(temp * reciprocal_m));
//   }

//   Seed[n] = seed;
// }



/********************************************
*********************************************
        FUNCTIONS FOR THE EOM-S
*********************************************
*********************************************/

/********************************************
* Basic spatial derivates of the PHASE FIELD:
********************************************/
double D_PHI_X(double PHI_XP, double PHI_XM){

 return (PHI_XP-PHI_XM)/(2.0*DX);
}

/**--------------------------------------------------------*/

double D_PHI_Y(double PHI_YP, double PHI_YM){

  return (PHI_YP-PHI_YM)/(2.0*DX);
}

/**--------------------------------------------------------*/

double D_PHI_X_FEL(double PHI_XP, double PHI){

 return (PHI_XP-PHI)/(DX);
}

/**--------------------------------------------------------*/

double D_PHI_Y_FEL(double PHI_YP, double PHI){

 return (PHI_YP-PHI)/(DX);
}

/*******************************************
    Functions for the PHASE FIELD EOM:
*******************************************/

double THETA_FUN(double PHI_XP,double  PHI_XM,double  PHI_YP,double  PHI_YM){

 return atan2(D_PHI_Y(PHI_YP,PHI_YM),D_PHI_X(PHI_XP,PHI_XM));
}

/**--------------------------------------------------------*/

double THETA_FUN_FEL_X(double PHI, double PHI_XP,double PHI_XPYP,double PHI_XPYM,double PHI_YP,double PHI_YM){

 double y=0.5*(D_PHI_Y(PHI_XPYP,PHI_XPYM)+D_PHI_Y(PHI_YP,PHI_YM));
 double x=D_PHI_X_FEL(PHI_XP,PHI );

 return atan2(y,x);
}

/**--------------------------------------------------------*/

double THETA_FUN_FEL_Y(double PHI, double PHI_YP, double PHI_XPYP, double PHI_XMYP, double PHI_XP, double PHI_XM){

 double y=D_PHI_Y_FEL(PHI_YP,PHI);
 double x=0.5*(D_PHI_X(PHI_XP,PHI_XM)+D_PHI_X(PHI_XPYP,PHI_XMYP));

 return atan2(y,x);
}

/**--------------------------------------------------------*/

double SURFACE_ANISOTROPY_FUN(double PHI_XP,double PHI_XM,double PHI_YP,double PHI_YM, double ORI){

 return 1+S_0_SURFACE*cos(K*THETA_FUN(PHI_XP,PHI_XM,PHI_YP,PHI_YM)-2.0*PI*ORI);
}

double KINETIC_ANISOTROPY_FUN(double PHI_XP,double PHI_XM,double PHI_YP,double PHI_YM, double ORI){

 return 1+S_0_KINETIC*cos(K*THETA_FUN(PHI_XP,PHI_XM,PHI_YP,PHI_YM)-2.0*PI*ORI);
}

/**--------------------------------------------------------*/

double SURFACE_ANISOTROPY_FUN_FEL_X(double PHI, double PHI_XP,double PHI_XPYP,double PHI_XPYM,double PHI_YP,double PHI_YM, double ORI, double ORI_XP){

 return 1+S_0_SURFACE*cos(K*THETA_FUN_FEL_X(PHI,PHI_XP,PHI_XPYP,PHI_XPYM,PHI_YP,PHI_YM)-PI*(ORI+ORI_XP));
}

/**--------------------------------------------------------*/

double SURFACE_ANISOTROPY_FUN_FEL_Y(double PHI, double PHI_YP, double PHI_XPYP, double PHI_XMYP, double PHI_XP, double PHI_XM, double ORI, double ORI_YP){

return 1+S_0_SURFACE*cos(K*THETA_FUN_FEL_Y(PHI,PHI_YP,PHI_XPYP,PHI_XMYP,PHI_XP,PHI_XM)-PI*(ORI+ORI_YP));
}

/**--------------------------------------------------------*/

double Laplace(double PHI, double PHI_XP,double PHI_XM,double PHI_YP,double PHI_YM){

 return (PHI_XP+PHI_XM+PHI_YP+PHI_YM-4.0*PHI)/(DX*DX);
}

/**--------------------------------------------------------*/
// g(phi)=(1/4)*phi**2*(1-phi)**2
double G_PHI(double PHI){

  return 0.25*PHI*PHI*(1.0-PHI)*(1.0-PHI);
}

/**--------------------------------------------------------*/
// g'(phi)=(1/2)*phi*(1-phi)*(1-2*phi)
double D_G_PHI(double PHI){

  return 0.5*PHI*(1.0-PHI)*(1.0-2.0*PHI);
}

/**
* TRASITION FUNCTION: P(PHI)=10*PHI**3*(10-15*PHI+6*PHI**2)
*/
double P_PHI(double PHI){

 if(PHI<0) return 0;
 else if(PHI>1) return 1;
 else return PHI*PHI*PHI*(10.0-15.0*PHI+6.0*PHI*PHI);
}

/**
* CALCULATING TRANSITION FUNCTION FOR THE DIVERGENCE TERM
*/
//p(phi[i+1/2,j])
double P_PHI_XPF(double PHI, double PHI_XP){

 double PHI_XPF=0.5*(PHI+PHI_XP);

 if(PHI_XPF<0) return 0;
 else if(PHI_XPF>1) return 1;
 else return PHI_XPF*PHI_XPF*PHI_XPF*(10.0-15.0*PHI_XPF+6.0*PHI_XPF*PHI_XPF);

}

/**--------------------------------------------------------*/
//p(phi[i-1/2,j])
double P_PHI_XMF(double PHI,double PHI_XM){

 double PHI_XMF=0.5*(PHI+PHI_XM);

 if(PHI_XMF<0) return 0;
 else if(PHI_XMF>1) return 1;
 else return PHI_XMF*PHI_XMF*PHI_XMF*(10.0-15.0*PHI_XMF+6.0*PHI_XMF*PHI_XMF);
}

/**--------------------------------------------------------*/
//p(phi[i,j+1/2])
double P_PHI_YPF(double PHI,double PHI_YP){

 double PHI_YPF=0.5*(PHI+PHI_YP);

 if(PHI_YPF<0) return 0;
 else if(PHI_YPF>1) return 1;
 else return PHI_YPF*PHI_YPF*PHI_YPF*(10.0-15.0*PHI_YPF+6.0*PHI_YPF*PHI_YPF);
}

/**--------------------------------------------------------*/
//p(phi[i,j-1/2])
double P_PHI_YMF(double PHI, double PHI_YM){

 double PHI_YMF=0.5*(PHI+PHI_YM);

 if(PHI_YMF<0) return 0;
 else if(PHI_YMF>1) return 1;
 else return PHI_YMF*PHI_YMF*PHI_YMF*(10.0-15.0*PHI_YMF+6.0*PHI_YMF*PHI_YMF);
}

/**--------------------------------------------------------*/
//p'(phi[i,j])
double D_P_PHI(double PHI){

  if(0.0<PHI && PHI<1.0) return 30.0*PHI*PHI-60.0*PHI*PHI*PHI+30.0*PHI*PHI*PHI*PHI;
  else return 0;
}

/**--------------------------------------------------------*/
// thermodinamic driving force term in the phase field EOM:
double DF(double CONC ){
  return DLESS_DH_F_CU_PF*CONC*(1.0-T_0/T_M_CU)+DLESS_DH_F_NI_PF*(1.0-CONC)*(1.0-T_0/T_M_NI);
//  double p1=DLESS_DH_F_CU_PF;
//  double p2=CONC;
//  double p3=(1.0-T/T_M_CU);
//  double p4=DLESS_DH_F_NI_PF;
//  double p5=(1.0-CONC);
//  double p6=(1.0-T/T_M_NI);

//  return DLESS_DH_F_CU_PF*CONC+DLESS_DH_F_NI_PF*(1.0-CONC);


}

/**--------------------------------------------------------*/
// Orientation field term in the phase field EOM:
double ABSGRAD_THETA(double ORI,double ORI_XP,double ORI_XM,double ORI_YP,double ORI_YM){

 double GRAD_XPF, GRAD_XMF, GRAD_YPF, GRAD_YMF;
 double GRAD_X, GRAD_Y;


 GRAD_XPF=ANGDIFF(ORI_XP,ORI)/DX;
 GRAD_XMF=ANGDIFF(ORI,ORI_XM)/DX;

 GRAD_YPF=ANGDIFF(ORI_YP,ORI)/DX;
 GRAD_YMF=ANGDIFF(ORI,ORI_YM)/DX;

 GRAD_X=0.5*(GRAD_XPF+GRAD_XMF);
 GRAD_Y=0.5*(GRAD_YPF+GRAD_YMF);

 return sqrt(GRAD_X*GRAD_X+GRAD_Y*GRAD_Y);
}

double F_ORI(double ORI,double ORI_XP,double ORI_XM,double ORI_YP,double ORI_YM){

 double GRAD_XPF, GRAD_XMF, GRAD_YPF, GRAD_YMF;
 double GRAD_X, GRAD_Y;

 GRAD_XPF=ANGDIFF(ORI_XP,ORI);
 GRAD_XMF=ANGDIFF(ORI,ORI_XM);

 GRAD_YPF=ANGDIFF(ORI_YP,ORI);
 GRAD_YMF=ANGDIFF(ORI,ORI_YM);

 GRAD_X=0.5*(GRAD_XPF+GRAD_XMF);
 GRAD_Y=0.5*(GRAD_YPF+GRAD_YMF);

 double absgrad_theta=sqrt(GRAD_X*GRAD_X+GRAD_Y*GRAD_Y);

 double F0,F1;
 if(absgrad_theta<3.0/(4.0*FORI_M)) F0=fabs(sin(2.0*PI*FORI_M*absgrad_theta));
 else F0=1.0;

 if(absgrad_theta<1.0/(4.0*FORI_N)) F1=fabs(sin(2.0*PI*FORI_N*absgrad_theta));
 else F1=1.0;

 return 1.0/(2.0*DX)*(FORI_X*F0+(1.0-FORI_X)*F1);
}


/**--------------------------------------------------------*/

double ABSGRAD2_THETA(double ORI,double ORI_XP,double ORI_XM,double ORI_YP,double ORI_YM){

 double grad_xpf, grad_xmf, grad_ypf, grad_ymf;

 grad_xpf=ANGDIFF(ORI_XP,ORI);
 grad_xmf=ANGDIFF(ORI,ORI_XM);
 grad_ymf=ANGDIFF(ORI_YP,ORI);
 grad_ypf=ANGDIFF(ORI,ORI_YM);

 double grad_x, grad_y;

 grad_x=(grad_xpf+grad_xmf)/(2.0*DX);
 grad_y=(grad_ypf+grad_ymf)/(2.0*DX);

 return (grad_x*grad_x+grad_y*grad_y);
}

/**--------------------------------------------------------*/
// Vector order parameter
double ABSGRAD_THETA_VECTOR(double P, double P_XP, double P_XM, double P_YP, double P_YM,
  double Q, double Q_XP, double Q_XM, double Q_YP, double Q_YM){

  double grad_px=(P_XP-P_XM)/(2.0*DX);
  double grad_py=(P_YP-P_YM)/(2.0*DX);

  double grad_qx=(Q_XP-Q_XM)/(2.0*DX);
  double grad_qy=(Q_YP-Q_YM)/(2.0*DX);

  double grad_square=grad_px*grad_px+grad_py*grad_py+grad_qx*grad_qx+grad_qy*grad_qy;

  return sqrt(grad_square);
}


/**--------------------------------------------------------*/

double D_Q_PHI(double PHI){

 double a=(1.0-PHI);
 double b=(7.0*PHI*PHI*PHI-6*PHI*PHI*PHI*PHI);
 double c=(21.0*PHI*PHI-24.0*PHI*PHI*PHI);

// if there is numerical instability, so that the denominator becames zero, than the function should return with zero:
 double delta=0.000000;
 if (delta<(a*a*a) && 0<PHI && PHI<1) return (c*a+3*b)/(a*a*a*a);
 else return 0;

}

/**--------------------------------------------------------*/

double ANISOTROPY_TERM(double PHI, double PHI_XP,double PHI_XM,double PHI_YP,double PHI_YM,double PHI_XPYP,
                        double PHI_XPYM,double PHI_XMYP,double PHI_XMYM,double ORI,double ORI_XP,double ORI_XM,
                        double ORI_YP,double ORI_YM){

   double S0_FUN_FELX =    SURFACE_ANISOTROPY_FUN_FEL_X(PHI,    PHI_XP, PHI_XPYP, PHI_XPYM, PHI_YP,   PHI_YM,   ORI,    ORI_XP);
   double S0_FUN_FELX_XM = SURFACE_ANISOTROPY_FUN_FEL_X(PHI_XM, PHI,    PHI_YP,   PHI_YM,   PHI_XMYP, PHI_XMYM, ORI_XM, ORI);

   double S0_FUN_FELY =    SURFACE_ANISOTROPY_FUN_FEL_Y(PHI,    PHI_YP, PHI_XPYP, PHI_XMYP, PHI_XP,   PHI_XM,   ORI,    ORI_YP);
   double S0_FUN_FELY_YM = SURFACE_ANISOTROPY_FUN_FEL_Y(PHI_YM, PHI,    PHI_XP,   PHI_XM,   PHI_XPYM, PHI_XMYM, ORI_YM, ORI);

   double THETA_XPF=THETA_FUN_FEL_X( PHI,    PHI_XP, PHI_XPYP, PHI_XPYM, PHI_YP,   PHI_YM);
   double THETA_XMF=THETA_FUN_FEL_X( PHI_XM, PHI,    PHI_YP,   PHI_YM,   PHI_XMYP, PHI_XMYM );

   double THETA_YPF=THETA_FUN_FEL_Y( PHI,    PHI_YP, PHI_XPYP, PHI_XMYP, PHI_XP,   PHI_XM);
   double THETA_YMF=THETA_FUN_FEL_Y( PHI_YM, PHI,    PHI_XP,   PHI_XM,   PHI_XPYM, PHI_XMYM);

   // Terms of the expansion of the anisotropy at the PF EOM:
   // part_x of the divergence:
   double T1= S0_FUN_FELX*  K*S_0_SURFACE*sin(K*THETA_XPF-PI*(ORI_XP+ORI))*
             (D_PHI_Y( PHI_XPYP, PHI_XPYM)+D_PHI_Y( PHI_YP, PHI_YM))/2.0;

   //every argument in T1 is shifted one step in the negative direction at the x-axis:
   double T2= S0_FUN_FELX_XM*K*S_0_SURFACE*sin(K*THETA_XMF-PI*(ORI+ORI_XM))*
             (D_PHI_Y( PHI_YP, PHI_YM )+D_PHI_Y( PHI_XMYP, PHI_XMYM ))/2.0;


   double T3= S0_FUN_FELX*S0_FUN_FELX*D_PHI_X_FEL( PHI_XP, PHI);

   double T4= S0_FUN_FELX_XM*S0_FUN_FELX_XM*D_PHI_X_FEL(PHI,PHI_XM);

   double PART1=(T1-T2)/DX;
   double PART2=(T3-T4)/DX;

  // part_y of the divergence:
   double T5= S0_FUN_FELY*K*S_0_SURFACE*sin(K*THETA_YPF-PI*(ORI_YP+ORI))*
             (D_PHI_X( PHI_XPYP, PHI_XMYP)+D_PHI_X( PHI_XP, PHI_XM))/2.0;

   double T6= S0_FUN_FELY_YM*K*S_0_SURFACE*sin(K*THETA_YMF-PI*(ORI_YM+ORI))*
             (D_PHI_X(PHI_XPYM,PHI_XMYM)+D_PHI_X(PHI_XP,PHI_XM))/2.0;

   double T7= S0_FUN_FELY*S0_FUN_FELY*D_PHI_Y_FEL(PHI_YP, PHI);

   double T8= S0_FUN_FELY_YM*S0_FUN_FELY_YM*D_PHI_Y_FEL(PHI,PHI_YM);

   double PART3=-(T5-T6)/DX;
   double PART4= (T7-T8)/DX;

   return PART1+PART2+PART3+PART4;
}


/**
* PHASE FIELD UPDATE FUNCTION:
*/
__kernel void phase_field_update(__global double* Phi,
                                 __global double* C,
                                 __global double* PhiNew,
                                 __global double* Ori,
                                 __global mwc64x_state_t* RandState){
 int n=get_global_id(0);

 double c=C[n];
 double tmp;
 double phidot=0.0;


  // Calculate first neighbours' index:
  #if BCTYPE_X==PERIODIC
    int XP = ( MYDIV(n,YSTEP) == MYDIV(n+XSTEP,YSTEP) ? n+XSTEP : n+XSTEP-YSTEP );
    int XM = ( MYDIV(n,YSTEP) == MYDIV(n-XSTEP,YSTEP) ? n-XSTEP : n-XSTEP+YSTEP );

    int XPYP = ( MYDIV(XP, LINSIZE) == MYDIV(XP+YSTEP, LINSIZE) ? XP+YSTEP : XP+YSTEP-LINSIZE );
    int XPYM = ( MYDIV(XP, LINSIZE) == MYDIV(XP-YSTEP, LINSIZE) ? XP-YSTEP : XP-YSTEP+LINSIZE );
    int XMYP = ( MYDIV(XM, LINSIZE) == MYDIV(XM+YSTEP, LINSIZE) ? XM+YSTEP : XM+YSTEP-LINSIZE );
    int XMYM = ( MYDIV(XM, LINSIZE) == MYDIV(XM-YSTEP, LINSIZE) ? XM-YSTEP : XM-YSTEP+LINSIZE );
  #elif BCTYPE_X==NOFLUX
    int XP = ( MYDIV(n,YSTEP) == MYDIV(n+XSTEP,YSTEP) ? n+XSTEP : n );
    int XM = ( MYDIV(n,YSTEP) == MYDIV(n-XSTEP,YSTEP) ? n-XSTEP : n );

    int XPYP = ( MYDIV(XP, LINSIZE) == MYDIV(XP+YSTEP, LINSIZE) ? XP+YSTEP : XP );
    int XPYM = ( MYDIV(XP, LINSIZE) == MYDIV(XP-YSTEP, LINSIZE) ? XP-YSTEP : XP );
    int XMYP = ( MYDIV(XM, LINSIZE) == MYDIV(XM+YSTEP, LINSIZE) ? XM+YSTEP : XM );
    int XMYM = ( MYDIV(XM, LINSIZE) == MYDIV(XM-YSTEP, LINSIZE) ? XM-YSTEP : XM );
  #endif

  #if DIMENSION==2
  #if BCTYPE_Y==PERIODIC
    int YP = ( MYDIV(n, LINSIZE) == MYDIV(n+YSTEP, LINSIZE) ? n+YSTEP : n+YSTEP-LINSIZE );
    int YM = ( MYDIV(n, LINSIZE) == MYDIV(n-YSTEP, LINSIZE) ? n-YSTEP : n-YSTEP+LINSIZE );
  #elif BCTYPE_Y==NOFLUX
    int YP = ( MYDIV(n, LINSIZE) == MYDIV(n+YSTEP, LINSIZE) ? n+YSTEP : n );
    int YM = ( MYDIV(n, LINSIZE) == MYDIV(n-YSTEP, LINSIZE) ? n-YSTEP : n );
  #endif
  #endif

  #if DIMENSION==3
  #if BCTYPE_Y==PERIODIC
    int YP = ( MYDIV(n,ZSTEP) == MYDIV(n+YSTEP,ZSTEP) ? n+YSTEP : n+YSTEP-ZSTEP );
    int YM = ( MYDIV(n,ZSTEP) == MYDIV(n-YSTEP,ZSTEP) ? n-YSTEP : n-YSTEP+ZSTEP );
  #elif BCTYPE_Y==NOFLUX
    int YP = ( MYDIV(n,ZSTEP) == MYDIV(n+YSTEP,ZSTEP) ? n+YSTEP : n );
    int YM = ( MYDIV(n,ZSTEP) == MYDIV(n-YSTEP,ZSTEP) ? n-YSTEP : n );
  #endif
  #if BCTYPE_Z==PERIODIC
    int ZP = ( MYDIV(n, LINSIZE) == MYDIV(n+ZSTEP, LINSIZE) ? n+ZSTEP : n+ZSTEP-LINSIZE );
    int ZM = ( MYDIV(n, LINSIZE) == MYDIV(n-ZSTEP, LINSIZE) ? n-ZSTEP : n-ZSTEP+LINSIZE );
  #elif BCTYPE_Z==NOFLUX
    int ZP = ( MYDIV(n, LINSIZE) == MYDIV(n+ZSTEP, LINSIZE) ? n+ZSTEP : n );
    int ZM = ( MYDIV(n, LINSIZE) == MYDIV(n-ZSTEP, LINSIZE) ? n-ZSTEP : n );
  #endif
  #endif

 double phi=Phi[n];
 double phi_xp=Phi[XP];
 double phi_xm=Phi[XM];
 double phi_yp=Phi[YP];
 double phi_ym=Phi[YM];
 double phi_xpyp=Phi[XPYP];
 double phi_xpym=Phi[XPYM];
 double phi_xmyp=Phi[XMYP];
 double phi_xmym=Phi[XMYM];

 double ori=Ori[n];
 double ori_xp=Ori[XP];
 double ori_xm=Ori[XM];
 double ori_yp=Ori[YP];
 double ori_ym=Ori[YM];

 double mphi=DLESS_M_PHI;

 switch(ANISOTROPY){
  case(1):
  tmp=(phi_xp+phi_xm+phi_yp+phi_ym-4.0*phi)/(DX*DX);
  tmp=Laplace(phi, phi_xp, phi_xm, phi_yp, phi_ym);
  phidot+=tmp;
  break;

  case(2):
   mphi*=KINETIC_ANISOTROPY_FUN(phi_xp, phi_xm, phi_yp, phi_ym, ori);
   //tmp=Laplace(phi, phi_xp, phi_xm, phi_yp, phi_ym);
   tmp=(phi_xp+phi_xm+phi_yp+phi_ym-4.0*phi)/(DX*DX);
   phidot+=tmp;
   break;

 case(3):
  tmp=ANISOTROPY_TERM( phi,  phi_xp, phi_xm, phi_yp, phi_ym, phi_xpyp,
                         phi_xpym, phi_xmyp, phi_xmym, ori, ori_xp, ori_xm,
                         ori_yp, ori_ym);
  phidot=phidot+tmp;
 break;
 }



 tmp=-DLESS_W*D_G_PHI(phi);
 phidot=phidot+tmp;

 tmp=D_P_PHI(phi)*DF(c);
 phidot=phidot+tmp;

// here there is a switch, where according to the model we would like to simulate, the orientation term changes:
 switch(MODELL)
 {
     case 1: //this is the maggrad-model
     tmp=-D_P_PHI(phi)*(DLESS_HT*ABSGRAD_THETA( ori, ori_xp, ori_xm, ori_yp, ori_ym));
     phidot=phidot+tmp;
     break;

     case 2: // this is the modified Plapp-model:
     tmp=-D_Q_PHI(phi)*(DLESS_HT_PLAPP*ABSGRAD2_THETA( ori, ori_xp, ori_xm, ori_yp, ori_ym));
     //tmp=0.0;
     phidot=phidot+tmp;
     break;

     case 3: // this is the sidebranching case:
     tmp=-D_P_PHI(phi)*(DLESS_HT*F_ORI( ori, ori_xp, ori_xm, ori_yp, ori_ym));
     phidot=phidot+tmp;
     break;

     case 4: //this is the maggrad-model, but with phi**3 coefficient function
     tmp=-3.0*phi*phi*(DLESS_HT*ABSGRAD_THETA( ori, ori_xp, ori_xm, ori_yp, ori_ym));
     phidot=phidot+tmp;
     break;
 }


//multiply the right hand side with the mobility
 phidot*=mphi; // Noise sample

 double noise;
 mwc64x_state_t rng = RandState[n];

 // Normal noise
 noise = PHASE_NOISE_AMPLITUDE*random_normal(&rng)*(1.0-P_PHI(phi))*sqrt(ADT);
 //noise = random_normal(n, &seed, PHASE_NOISE_AMPLITUDE)*(1.0-P_PHI(phi))*sqrt(ADT);//*sqrt(ADT);

 // Save PRNG state
 RandState[n] = rng;

 if (phi+(phidot)*DT<-0.1 || phi+(phidot)*DT>1.0) PhiNew[n]=0.99;
 else PhiNew[n]=phi+(phidot)*DT+noise;

}


/****************************************
  FUNCTIONS FOR CONCENTRATION FIELD EOM:
*****************************************/
// inside the gradient:
double INNER_FUNCTION(double phi, double conc){

 double DH_CU=DLESS_DH_F_CU_CF*(1.0-T_0/T_M_CU);
 double DH_NI=DLESS_DH_F_NI_CF*(1.0-T_0/T_M_NI);

 return (1.0-P_PHI(phi))*(DH_CU-DH_NI)+log(conc/(1.0-conc));
}

/**--------------------------------------------------------*/
// outside the gradient but inside the divergence:
double OUTER_FUNCTION(double phi, double conc){

 return (D_S_RAT+(1.0-D_S_RAT)*(1.0-P_PHI(phi)))*conc*(1.0-conc);
}

/**--------------------------------------------------------*/
// concentration current in the x direction:
__kernel void J_X_CALC(__global double* PHI,__global double* Jx,__global double* C){
 int n=get_global_id(0);

  #if BCTYPE_X==PERIODIC
  int XM = ( MYDIV(n,YSTEP) == MYDIV(n-XSTEP,YSTEP) ? n-XSTEP : n-XSTEP+YSTEP );
  #elif BCTYPE_X==NOFLUX
  int XM = ( MYDIV(n,YSTEP) == MYDIV(n-XSTEP,YSTEP) ? n-XSTEP : n );
  #endif

 double phi=PHI[n];
 double phi_xm=PHI[XM];


 double conc=C[n];
 double conc_xm=C[XM];

 double out_xmf=(OUTER_FUNCTION(phi_xm,conc_xm)+OUTER_FUNCTION(phi,conc))*0.5;
 double in_xmf =(INNER_FUNCTION(phi,conc)-INNER_FUNCTION(phi_xm,conc_xm))/DX;

 Jx[n]=out_xmf*in_xmf;
}

/**--------------------------------------------------------*/
// concentration current in the y direction:
__kernel void J_Y_CALC(__global double* PHI,__global double* Jy,__global double* C){
 int n=get_global_id(0);

  #if BCTYPE_X==PERIODIC
  int YM = ( MYDIV(n, LINSIZE) == MYDIV(n-YSTEP, LINSIZE) ? n-YSTEP : n-YSTEP+LINSIZE );
 #elif BCTYPE_X==NOFLUX
  int YM = ( MYDIV(n, LINSIZE) == MYDIV(n-YSTEP, LINSIZE) ? n-YSTEP : n );
 #endif

 double phi=PHI[n];
 double phi_ym=PHI[YM];

 double conc=C[n];
 double conc_ym=C[YM];

 double out_ymf=(OUTER_FUNCTION(phi_ym,conc_ym)+OUTER_FUNCTION(phi,conc))*0.5;
 double in_ymf =(INNER_FUNCTION(phi,conc)-INNER_FUNCTION(phi_ym,conc_ym))/DX;

 Jy[n]=out_ymf*in_ymf;
}

/**--------------------------------------------------------*/

__kernel void conc_field_update(__global double* PHI, __global double* C, __global double* C_new, __global double* Jx, __global double* Jy){
 int n=get_global_id(0);
 #if BCTYPE_X==PERIODIC
  int XP = ( MYDIV(n,YSTEP) == MYDIV(n+XSTEP,YSTEP) ? n+XSTEP : n+XSTEP-YSTEP );

  int YP = ( MYDIV(n, LINSIZE) == MYDIV(n+YSTEP, LINSIZE) ? n+YSTEP : n+YSTEP-LINSIZE );
 #elif BCTYPE_X==NOFLUX
  int XP = ( MYDIV(n,YSTEP) == MYDIV(n+XSTEP,YSTEP) ? n+XSTEP : n );

  int YP = ( MYDIV(n, LINSIZE) == MYDIV(n+YSTEP, LINSIZE) ? n+YSTEP : n );
  #endif

 double c_dot=0.0;

 c_dot+=(Jx[XP]-Jx[n])/DX+(Jy[YP]-Jy[n])/DX;

 C_new[n]=C[n]+c_dot*DT;

}


/****************************************
  FUNCTIONS FOR ORIENTATION FIELD EOM:
*****************************************/

double Q_PHI(double PHI){

// double a=(1.0-PHI);

 if (0.0<PHI && PHI<1.0 && (1.0-PHI)*(1.0-PHI)*(1.0-PHI)!=0) return (7.0*PHI*PHI*PHI-6.0*PHI*PHI*PHI*PHI)/((1.0-PHI)*(1.0-PHI)*(1.0-PHI));
 else return 0.0;

}

__kernel void orientation_field_update(__global double* PHI, 
                                       __global double* Ori, 
                                       __global double* Ori_new,
                                       __global mwc64x_state_t* RandState){
 int n=get_global_id(0);

 #if BCTYPE_X==PERIODIC
  int XP = ( MYDIV(n,YSTEP) == MYDIV(n+XSTEP,YSTEP) ? n+XSTEP : n+XSTEP-YSTEP );
  int XM = ( MYDIV(n,YSTEP) == MYDIV(n-XSTEP,YSTEP) ? n-XSTEP : n-XSTEP+YSTEP );

  int XPYP = ( MYDIV(XP, LINSIZE) == MYDIV(XP+YSTEP, LINSIZE) ? XP+YSTEP : XP+YSTEP-LINSIZE );
  int XPYM = ( MYDIV(XP, LINSIZE) == MYDIV(XP-YSTEP, LINSIZE) ? XP-YSTEP : XP-YSTEP+LINSIZE );
  int XMYP = ( MYDIV(XM, LINSIZE) == MYDIV(XM+YSTEP, LINSIZE) ? XM+YSTEP : XM+YSTEP-LINSIZE );
  int XMYM = ( MYDIV(XM, LINSIZE) == MYDIV(XM-YSTEP, LINSIZE) ? XM-YSTEP : XM-YSTEP+LINSIZE );

  int YP = ( MYDIV(n, LINSIZE) == MYDIV(n+YSTEP, LINSIZE) ? n+YSTEP : n+YSTEP-LINSIZE );
  int YM = ( MYDIV(n, LINSIZE) == MYDIV(n-YSTEP, LINSIZE) ? n-YSTEP : n-YSTEP+LINSIZE );
 #elif BCTYPE_X==NOFLUX
  int XP = ( MYDIV(n,YSTEP) == MYDIV(n+XSTEP,YSTEP) ? n+XSTEP : n );
  int XM = ( MYDIV(n,YSTEP) == MYDIV(n-XSTEP,YSTEP) ? n-XSTEP : n );

  int XPYP = ( MYDIV(XP, LINSIZE) == MYDIV(XP+YSTEP, LINSIZE) ? XP+YSTEP : XP );
  int XPYM = ( MYDIV(XP, LINSIZE) == MYDIV(XP-YSTEP, LINSIZE) ? XP-YSTEP : XP );
  int XMYP = ( MYDIV(XM, LINSIZE) == MYDIV(XM+YSTEP, LINSIZE) ? XM+YSTEP : XM );
  int XMYM = ( MYDIV(XM, LINSIZE) == MYDIV(XM-YSTEP, LINSIZE) ? XM-YSTEP : XM );

  int YP = ( MYDIV(n, LINSIZE) == MYDIV(n+YSTEP, LINSIZE) ? n+YSTEP : n );
  int YM = ( MYDIV(n, LINSIZE) == MYDIV(n-YSTEP, LINSIZE) ? n-YSTEP : n );
 #endif

 double ori=Ori[n];
 double ori_xp=Ori[XP];
 double ori_xm=Ori[XM];
 double ori_yp=Ori[YP];
 double ori_ym=Ori[YM];
 double ori_xpyp=Ori[XPYP];
 double ori_xpym=Ori[XPYM];
 double ori_xmyp=Ori[XMYP];
 double ori_xmym=Ori[XMYM];

 double phi=PHI[n];
 double phi_xp=PHI[XP];
 double phi_xm=PHI[XM];
 double phi_yp=PHI[YP];
 double phi_ym=PHI[YM];

 double phi_xpyp=PHI[XPYP];
 double phi_xpym=PHI[XPYM];
 double phi_xmyp=PHI[XMYP];
 double phi_xmym=PHI[XMYM];

 double grad_xpf, grad_xmf, grad_ypf, grad_ymf;

 grad_xpf=ANGDIFF(ori_xp,ori);
 grad_xmf=ANGDIFF(ori,ori_xm);

 grad_ypf=ANGDIFF(ori_yp,ori);
 grad_ymf=ANGDIFF(ori,ori_ym);

 double oridot;

 double p_xpf, p_xmf, p_ypf, p_ymf;
 double part_x, part_y;
 double phi_xpf, phi_xmf, phi_ypf, phi_ymf;
 double qfi_xpf, qfi_xmf, qfi_ypf, qfi_ymf;

 double phi_xpf_ypf,phi_xmf_ymf,phi_xpf_ymf,phi_xmf_ypf;
 double qfi_xpf_ypf,qfi_xmf_ymf,qfi_xpf_ymf,qfi_xmf_ypf;

 double grad_y_xpf, grad_y_xmf, grad_x_ypf, grad_x_ymf;
 double abs_grad_xpf, abs_grad_xmf, abs_grad_ypf, abs_grad_ymf;
 double ref=1.0e-9;
 double px_p, px_m, py_p, py_m;
 double divgrad;

 // new terms necessery for the 9 point Laplacian
 double grad_xy_xpf_ypf,grad_xy_xmf_ymf,grad_xy_xmf_ypf,grad_xy_xpf_ymf;

 grad_xy_xpf_ypf=ANGDIFF(ori_xpyp,ori);
 grad_xy_xmf_ymf=ANGDIFF(ori,ori_xmym);
 grad_xy_xpf_ymf=ANGDIFF(ori_xpym,ori);
 grad_xy_xmf_ypf=ANGDIFF(ori,ori_xmyp);

 double part_xy_plus, part_xy_minus;

 switch(MODELL)
 {
  case 1:
   // p_xpf=(phi+phi_xp)*(phi+phi_xp)*(phi+phi_xp)/8.0;
   // p_xmf=(phi+phi_xm)*(phi+phi_xm)*(phi+phi_xm)/8.0;

   // p_ypf=(phi+phi_yp)*(phi+phi_yp)*(phi+phi_yp)/8.0;
   // p_ymf=(phi+phi_ym)*(phi+phi_ym)*(phi+phi_ym)/8.0;
  
   p_xpf=P_PHI_XPF(phi,phi_xp);
   p_xmf=P_PHI_XMF(phi,phi_xm);

   p_ypf=P_PHI_YPF(phi,phi_yp);
   p_ymf=P_PHI_YMF(phi,phi_ym);


 /**
 xmyp|yp|xpyp
 ____|__|____
   xm|n |  xp
 ____|__|____
 xmym|ym|xmyp
     |  |
 */
   

   grad_x_ypf=(ANGDIFF(ori_xp,ori)+ANGDIFF(ori,ori_xm)+ANGDIFF(ori_xpyp,ori_yp)+ANGDIFF(ori_yp,ori_xmyp))/4.0;
   grad_x_ymf=(ANGDIFF(ori_xp,ori)+ANGDIFF(ori,ori_xm)+ANGDIFF(ori_xpym,ori_ym)+ANGDIFF(ori_ym,ori_xmym))/4.0;
   grad_y_xpf=(ANGDIFF(ori_yp,ori)+ANGDIFF(ori,ori_ym)+ANGDIFF(ori_xpyp,ori_xp)+ANGDIFF(ori_xp,ori_xpym))/4.0;
   grad_y_xmf=(ANGDIFF(ori_yp,ori)+ANGDIFF(ori,ori_ym)+ANGDIFF(ori_xmyp,ori_xm)+ANGDIFF(ori_xm,ori_xmym))/4.0;

   

   abs_grad_xpf=sqrt(grad_xpf*grad_xpf+grad_y_xpf*grad_y_xpf);
   abs_grad_xmf=sqrt(grad_xmf*grad_xmf+grad_y_xmf*grad_y_xmf);

   abs_grad_ypf=sqrt(grad_x_ypf*grad_x_ypf+grad_ypf*grad_ypf);
   abs_grad_ymf=sqrt(grad_x_ymf*grad_x_ymf+grad_ymf*grad_ymf);

   if(abs_grad_xpf<ref)    px_p=0.0; 
   else px_p=(p_xpf*grad_xpf/abs_grad_xpf);
   if(abs_grad_xmf<ref)    px_m=0.0; 
   else px_m=(p_xmf*grad_xmf/abs_grad_xmf);
   if(abs_grad_ypf<ref)    py_p=0.0; 
   else py_p=(p_ypf*grad_ypf/abs_grad_ypf);
   if(abs_grad_ymf<ref)    py_m=0.0; 
   else py_m=(p_ymf*grad_ymf/abs_grad_ymf);



   part_x=(px_p-px_m)/DX;
   part_y=(py_p-py_m)/DX;

   divgrad=part_x+part_y;

// deterministic term of the EOM:
   oridot=(DLESS_M_THETA_S+(DLESS_M_THETA_L-DLESS_M_THETA_S)*(1.0-P_PHI(phi)))*divgrad;
  break;

  case 2:
  // this is the gradtheta2 model:


   phi_xpf=(phi_xp+phi)/2.0;
   phi_xmf=(phi_xm+phi)/2.0;
   phi_ypf=(phi_yp+phi)/2.0;
   phi_ymf=(phi_ym+phi)/2.0;

   phi_xpf_ypf=(phi+phi_xpyp)/2.0;
   phi_xpf_ymf=(phi+phi_xpym)/2.0;
   phi_xmf_ypf=(phi+phi_xmyp)/2.0;
   phi_xmf_ymf=(phi+phi_xmym)/2.0;

   qfi_xpf=Q_PHI(phi_xpf);
   qfi_xmf=Q_PHI(phi_xmf);
   qfi_ypf=Q_PHI(phi_ypf);
   qfi_ymf=Q_PHI(phi_ymf);
   
   //since at model 1 we do not need the gradient, just the ANGDIFF, in this case we have to divide by DX:
   grad_xpf=grad_xpf/DX;
   grad_xmf=grad_xmf/DX;
   grad_ypf=grad_ypf/DX;
   grad_ymf=grad_ymf/DX;

   qfi_xpf_ypf=Q_PHI(phi_xpf_ypf);
   qfi_xpf_ymf=Q_PHI(phi_xpf_ymf);
   qfi_xmf_ypf=Q_PHI(phi_xmf_ypf);
   qfi_xmf_ymf=Q_PHI(phi_xmf_ymf);

   grad_xy_xpf_ypf=grad_xy_xpf_ypf/(2.0*DX);
   grad_xy_xpf_ymf=grad_xy_xpf_ymf/(2.0*DX);
   grad_xy_xmf_ypf=grad_xy_xmf_ypf/(2.0*DX);
   grad_xy_xmf_ymf=grad_xy_xmf_ymf/(2.0*DX);

   part_x=(qfi_xpf*grad_xpf-qfi_xmf*grad_xmf)/(DX);
   part_y=(qfi_ypf*grad_ypf-qfi_ymf*grad_ymf)/(DX);

   part_xy_plus= (qfi_xpf_ypf*grad_xy_xpf_ypf-qfi_xmf_ymf*grad_xy_xmf_ymf)/DX;
   part_xy_minus=(qfi_xpf_ymf*grad_xy_xpf_ymf-qfi_xmf_ypf*grad_xy_xmf_ypf)/DX;

   double mobility;
   mobility=(DLESS_M_THETA_S_PLAPP+(1.0-P_PHI(phi))*(DLESS_M_THETA_L_PLAPP-DLESS_M_THETA_S_PLAPP))*(1.0-phi)*(1.0-phi)*(1.0-phi);
   oridot=mobility*(2.0*(part_x+part_y)+(part_xy_plus+part_xy_minus))/3.0;

  break;

  case 3:
   p_xpf=P_PHI_XPF(phi,phi_xp);
   p_xmf=P_PHI_XMF(phi,phi_xm);

   p_ypf=P_PHI_YPF(phi,phi_yp);
   p_ymf=P_PHI_YMF(phi,phi_ym);

   grad_x_ypf=(ANGDIFF(ori_xp,ori)+ANGDIFF(ori,ori_xm)+ANGDIFF(ori_xpyp,ori_yp)+ANGDIFF(ori_yp,ori_xmyp))/4.0;
   grad_x_ymf=(ANGDIFF(ori_xp,ori)+ANGDIFF(ori,ori_xm)+ANGDIFF(ori_xpym,ori_ym)+ANGDIFF(ori_ym,ori_xmym))/4.0;
   grad_y_xpf=(ANGDIFF(ori_yp,ori)+ANGDIFF(ori,ori_ym)+ANGDIFF(ori_xpyp,ori_xp)+ANGDIFF(ori_xp,ori_xpym))/4.0;
   grad_y_xmf=(ANGDIFF(ori_yp,ori)+ANGDIFF(ori,ori_ym)+ANGDIFF(ori_xmyp,ori_xm)+ANGDIFF(ori_xm,ori_xmym))/4.0;

   abs_grad_xpf=sqrt(grad_xpf*grad_xpf+grad_y_xpf*grad_y_xpf);
   abs_grad_xmf=sqrt(grad_xmf*grad_xmf+grad_y_xmf*grad_y_xmf);

   abs_grad_ypf=sqrt(grad_x_ypf*grad_x_ypf+grad_ypf*grad_ypf);
   abs_grad_ymf=sqrt(grad_x_ymf*grad_x_ymf+grad_ymf*grad_ymf);

   ref=1.0e-9;

   if(abs_grad_xpf<ref)    px_p=0.0; 
   else px_p=(p_xpf*grad_xpf/abs_grad_xpf);
   if(abs_grad_xmf<ref)    px_m=0.0; 
   else px_m=(p_xmf*grad_xmf/abs_grad_xmf);
   if(abs_grad_ypf<ref)    py_p=0.0; 
   else py_p=(p_ypf*grad_ypf/abs_grad_ypf);
   if(abs_grad_ymf<ref)    py_m=0.0; 
   else py_m=(p_ymf*grad_ymf/abs_grad_ymf);

   double F0_xpf,F0_xmf,F0_ypf,F0_ymf;
   double F1_xpf,F1_xmf,F1_ypf,F1_ymf;
   
   F0_xpf=0.0;
   F0_xmf=0.0;
   F0_ypf=0.0;
   F0_ymf=0.0;
   
   F1_xpf=0.0;
   F1_xmf=0.0;
   F1_ypf=0.0;
   F1_ymf=0.0;
   

   if (abs_grad_xpf<3.0/(4.0*FORI_M)) F0_xpf=sign(sin(2.0*PI*FORI_M*abs_grad_xpf))*cos(2.0*PI*FORI_M*abs_grad_xpf);
   if (abs_grad_xmf<3.0/(4.0*FORI_M)) F0_xmf=sign(sin(2.0*PI*FORI_M*abs_grad_xmf))*cos(2.0*PI*FORI_M*abs_grad_xmf);
   if (abs_grad_ypf<3.0/(4.0*FORI_M)) F0_ypf=sign(sin(2.0*PI*FORI_M*abs_grad_ypf))*cos(2.0*PI*FORI_M*abs_grad_ypf);
   if (abs_grad_ymf<3.0/(4.0*FORI_M)) F0_ymf=sign(sin(2.0*PI*FORI_M*abs_grad_ymf))*cos(2.0*PI*FORI_M*abs_grad_ymf);

   if (abs_grad_xpf<1.0/(4.0*FORI_N)) F1_xpf=sign(sin(2.0*PI*FORI_N*abs_grad_xpf))*cos(2.0*PI*FORI_N*abs_grad_xpf);
   if (abs_grad_xmf<1.0/(4.0*FORI_N)) F1_xmf=sign(sin(2.0*PI*FORI_N*abs_grad_xmf))*cos(2.0*PI*FORI_N*abs_grad_xmf);
   if (abs_grad_ypf<1.0/(4.0*FORI_N)) F1_ypf=sign(sin(2.0*PI*FORI_N*abs_grad_ypf))*cos(2.0*PI*FORI_N*abs_grad_ypf);
   if (abs_grad_ymf<1.0/(4.0*FORI_N)) F1_ymf=sign(sin(2.0*PI*FORI_N*abs_grad_ymf))*cos(2.0*PI*FORI_N*abs_grad_ymf);



   part_x=(px_p*(FORI_X*F0_xpf*FORI_M+(1.0-FORI_X)*F1_xpf*FORI_N)*PI-px_m*(FORI_X*F0_xmf*FORI_M+(1.0-FORI_X)*F1_xmf*FORI_N)*PI)/DX;
   part_y=(py_p*(FORI_X*F0_ypf*FORI_M+(1.0-FORI_X)*F1_ypf*FORI_N)*PI-py_m*(FORI_X*F0_ymf*FORI_M+(1.0-FORI_X)*F1_ymf*FORI_N)*PI)/DX;

   divgrad=part_x+part_y;

// deterministic term of the EOM:
   oridot=(DLESS_M_THETA_S+(DLESS_M_THETA_L-DLESS_M_THETA_S)*(1.0-P_PHI(phi)))*divgrad;

  break;

  case 4:
   p_xpf=(phi+phi_xp)*(phi+phi_xp)*(phi+phi_xp)/8.0;
   p_xmf=(phi+phi_xm)*(phi+phi_xm)*(phi+phi_xm)/8.0;

   p_ypf=(phi+phi_yp)*(phi+phi_yp)*(phi+phi_yp)/8.0;
   p_ymf=(phi+phi_ym)*(phi+phi_ym)*(phi+phi_ym)/8.0;


   grad_x_ypf=(ANGDIFF(ori_xp,ori)+ANGDIFF(ori,ori_xm)+ANGDIFF(ori_xpyp,ori_yp)+ANGDIFF(ori_yp,ori_xmyp))/4.0;
   grad_x_ymf=(ANGDIFF(ori_xp,ori)+ANGDIFF(ori,ori_xm)+ANGDIFF(ori_xpym,ori_ym)+ANGDIFF(ori_ym,ori_xmym))/4.0;
   grad_y_xpf=(ANGDIFF(ori_yp,ori)+ANGDIFF(ori,ori_ym)+ANGDIFF(ori_xpyp,ori_xp)+ANGDIFF(ori_xp,ori_xpym))/4.0;
   grad_y_xmf=(ANGDIFF(ori_yp,ori)+ANGDIFF(ori,ori_ym)+ANGDIFF(ori_xmyp,ori_xm)+ANGDIFF(ori_xm,ori_xmym))/4.0;

   abs_grad_xpf=sqrt(grad_xpf*grad_xpf+grad_y_xpf*grad_y_xpf);
   abs_grad_xmf=sqrt(grad_xmf*grad_xmf+grad_y_xmf*grad_y_xmf);

   abs_grad_ypf=sqrt(grad_x_ypf*grad_x_ypf+grad_ypf*grad_ypf);
   abs_grad_ymf=sqrt(grad_x_ymf*grad_x_ymf+grad_ymf*grad_ymf);

   double ref=1.0e-9;

   if(abs_grad_xpf<ref)    px_p=0.0; 
   else px_p=(p_xpf*grad_xpf/abs_grad_xpf);
   if(abs_grad_xmf<ref)    px_m=0.0; 
   else px_m=(p_xmf*grad_xmf/abs_grad_xmf);
   if(abs_grad_ypf<ref)    py_p=0.0; 
   else py_p=(p_ypf*grad_ypf/abs_grad_ypf);
   if(abs_grad_ymf<ref)    py_m=0.0; 
   else py_m=(p_ymf*grad_ymf/abs_grad_ymf);



   part_x=(px_p-px_m)/DX;
   part_y=(py_p-py_m)/DX;

   divgrad=part_x+part_y;

// deterministic term of the EOM:
   oridot=(DLESS_M_THETA_S+(DLESS_M_THETA_L-DLESS_M_THETA_S)*(1.0-P_PHI(phi)))*divgrad;
  break;  
}
// noise term in the EOM:
 double noise;
 mwc64x_state_t rng = RandState[n];

 double pphi=(DLESS_M_THETA_S+(DLESS_M_THETA_L-DLESS_M_THETA_S)*(1.0-P_PHI(phi)))/DLESS_M_THETA_L;
 double pf=(1.0-phi);
 double pf10=pf*pf*pf*pf*pf*pf*pf*pf*pf*pf;
 noise = ORIENTATION_NOISE_AMPLITUDE*random_normal(&rng)*pphi*pf10*sqrt(ADT);
 //noise = random_normal(n, &seed, ORIENTATION_NOISE_AMPLITUDE)*pphi*pf10*sqrt(ADT); //sqrt(ADT)*
 RandState[n] = rng;
 double one=1.0;
 double mod1=fmod(Ori[n]+oridot*DT+noise,one);
 //double mod1=fmod(Ori[n]+oridot*DT,1.0);
 double new_value=fmod(mod1+one,one);
 Ori_new[n]=new_value;
}

/********************************************
*********************************************
   Initialize the orientation field with
          uniform random numbers:
*********************************************
*********************************************/


__kernel void initialize_orientation_field(__global double* Ori, __global mwc64x_state_t* RandState){
  int n=get_global_id(0);
  mwc64x_state_t rng = RandState[n];
  Ori[n]=random_uniform(&rng);
  //Ori[n]=0.25;
  RandState[n] = rng;
}

__kernel void initialize_phase_field(__global double* Phi){
 int n=get_global_id(0);
 Phi[n]=0;
}

__kernel void initialize_concentration_field(__global double* C){
 int n=get_global_id(0);
 C[n]=C_0;
}





__kernel void copy_fields(__global double* C, __global double* CN, __global double* Ori, __global double* OriN, __global double* Phi, __global double* PhiN){
 int n=get_global_id(0);

 CN[n]=C[n];
 OriN[n]=Ori[n];
 PhiN[n]=Phi[n];
}



<INIT_FUNCTIONS>
