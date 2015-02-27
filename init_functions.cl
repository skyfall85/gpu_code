__kernel void insert_one_nucleus(__global double* Phi, __global double* C,
 __global double* Ori, const double X0, const double Y0, const double radius, const double start_orientation,  __global mwc64x_state_t* RandState){
 
 int n=get_global_id(0);
 mwc64x_state_t rng = RandState[n];
 double ori_rand_term=random_uniform(&rng);
 RandState[n] = rng;
 int x,y;
 double rx2,ry2;
 double r;
 double phi=0.0;
 double c=0.0;
 double ori=0.0;
 double r_0=3.0*radius;
 double ori_ref;

 y = n/YSTEP;
 x = (n%YSTEP)/XSTEP;

 rx2=(x-X0)*(x-X0);
 ry2=(y-Y0)*(y-Y0);

 r=sqrt(rx2+ry2);


 if (r<r_0){
  phi=(1.0-tanh((r-r_0/2.0)*0.5))/2.0;
  ori_ref=(1.0-tanh((r-(3+r_0/2.0))*0.5))/2.0;
  c=C_0+(C_S-C_0)*phi;
  ori=start_orientation+(1.0-ori_ref)*ori_rand_term;
  //ori=start_orientation+(1.0-ori_ref)*(-0.25);


 Phi[n]=phi;
 C[n]=c;
 double mod1=fmod(ori,1.0);
 Ori[n]=fmod(mod1+1.0,1.0);;
 }


}

