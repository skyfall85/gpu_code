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

__kernel void insert_four_nucleus(__global double* Phi, __global double* C,
 __global double* Ori, const double radius, __global mwc64x_state_t* RandState){
 
 int n=get_global_id(0);
 mwc64x_state_t rng = RandState[n];
 double ori_rand_term=random_uniform(&rng);
 RandState[n] = rng;
 int x,y;
 double rx1_2,ry1_2, rx2_2,ry2_2,rx3_2,ry3_2,rx4_2,ry4_2;
 double r1,r2,r3,r4;
 double phi=0.0;
 double c=0.0;
 double ori=0.0;
 double r_0=3.0*radius;
 double ori_ref;

 y = n/YSTEP;
 x = (n%YSTEP)/XSTEP;

 double X1,X2,X3,X4;
 double Y1,Y2,Y3,Y4;

 X1=0; 
 Y1=0;
 X2=XSIZE;
 Y2=0;
 X3=0;
 Y3=YSIZE;
 X4=XSIZE;
 Y4=YSIZE;

 rx1_2=(x-X1)*(x-X1);
 ry1_2=(y-Y1)*(y-Y1);

 r1=sqrt(rx1_2+ry1_2);


 if (r1<r_0){
  phi=(1.0-tanh((r1-r_0/2.0)*0.5))/2.0;
  ori_ref=(1.0-tanh((r1-(3+r_0/2.0))*0.5))/2.0;
  c=C_0+(C_S-C_0)*phi;
  ori=0.1+(1.0-ori_ref)*ori_rand_term;
  
  Phi[n]=phi;
  C[n]=c;
  double mod1=fmod(ori,1.0);
  Ori[n]=fmod(mod1+1.0,1.0);;
 }

  /*--------------------------------------------------*/
  
 rx2_2=(x-X2)*(x-X2);
 ry2_2=(y-Y2)*(y-Y2);

 r2=sqrt(rx2_2+ry2_2);


 if (r2<r_0){
  phi=(1.0-tanh((r2-r_0/2.0)*0.5))/2.0;
  ori_ref=(1.0-tanh((r2-(3+r_0/2.0))*0.5))/2.0;
  c=C_0+(C_S-C_0)*phi;
  ori=0.35+(1.0-ori_ref)*ori_rand_term;
 
  Phi[n]=phi;
  C[n]=c;
  double mod1=fmod(ori,1.0);
  Ori[n]=fmod(mod1+1.0,1.0);;
  }

    /*--------------------------------------------------*/
  
 rx3_2=(x-X3)*(x-X3);
 ry3_2=(y-Y3)*(y-Y3);

 r3=sqrt(rx3_2+ry3_2);


 if (r3<r_0){
  phi=(1.0-tanh((r3-r_0/2.0)*0.5))/2.0;
  ori_ref=(1.0-tanh((r3-(3+r_0/2.0))*0.5))/2.0;
  c=C_0+(C_S-C_0)*phi;
  ori=0.65+(1.0-ori_ref)*ori_rand_term;
  //ori=start_orientation+(1.0-ori_ref)*(-0.25);  
  Phi[n]=phi;
  C[n]=c;
  double mod1=fmod(ori,1.0);
  Ori[n]=fmod(mod1+1.0,1.0);;
 }


  /*--------------------------------------------------*/
  
 rx4_2=(x-X4)*(x-X4);
 ry4_2=(y-Y4)*(y-Y4);

 r4=sqrt(rx4_2+ry4_2);


 if (r4<r_0){
  phi=(1.0-tanh((r4-r_0/2.0)*0.5))/2.0;
  ori_ref=(1.0-tanh((r4-(3+r_0/2.0))*0.5))/2.0;
  c=C_0+(C_S-C_0)*phi;
  ori=0.9+(1.0-ori_ref)*ori_rand_term;
  

 Phi[n]=phi;
 C[n]=c;
 double mod1=fmod(ori,1.0);
 Ori[n]=fmod(mod1+1.0,1.0);;
 }


}





__kernel void insert_four_rectangles(__global double* Phi, __global double* C,
 __global double* Ori){
 int n=get_global_id(0); 
 int x,y;
 double phi=0.0;
 double c=0.0;
 double ori=0.0;

 y = n/YSTEP;
 x = (n%YSTEP)/XSTEP;


 if (x<=XSIZE*0.5 && y<=YSIZE*0.5) ori=0.1;
 if (x<=XSIZE*0.5 && y>YSIZE*0.5) ori=0.35;
 if (x>XSIZE*0.5 && y<=YSIZE*0.5) ori=0.65;
 if (x>XSIZE*0.5 && y>YSIZE*0.5) ori=0.9;
 phi=0.7;
 c=C_0;

 Phi[n]=phi;
 C[n]=c;
 double mod1=fmod(ori,1.0);
 Ori[n]=fmod(mod1+1.0,1.0);
 
}


__kernel void insert_four_rectangles_cross(__global double* Phi, __global double* C,
 __global double* Ori){
 int n=get_global_id(0); 
 int x,y;
 double phi=0.0;
 double c=0.0;
 double ori=0.0;

 y = n/YSTEP;
 x = (n%YSTEP)/XSTEP;


 if (x<=y && y<=YSIZE-x) ori=0.1;
 if (x<=y && y>YSIZE-x) ori=0.35;
 if (x>y && y<=YSIZE-x) ori=0.65;
 if (x>y && y>YSIZE-x) ori=0.9;
 phi=0.95;
 c=C_0;

 Phi[n]=phi;
 C[n]=c;
 double mod1=fmod(ori,1.0);
 Ori[n]=fmod(mod1+1.0,1.0);
 
}





__kernel void insert_two_rectangles(__global double* Phi, __global double* C,
 __global double* Ori){
 int n=get_global_id(0); 
 int x,y;
 double phi=0.0;
 double c=0.0;
 double ori=0.0;

 y = n/YSTEP;
 x = (n%YSTEP)/XSTEP;


 if (x<=XSIZE*0.5) ori=0.3;
 if (x>XSIZE*0.5 ) ori=0.7;
 phi=0.7;
 c=C_0;

 Phi[n]=phi;
 C[n]=c;
 double mod1=fmod(ori,1.0);
 Ori[n]=fmod(mod1+1.0,1.0);
 
}

__kernel void insert_two_rectangles_noise_between(__global double* Phi, __global double* C,
 __global double* Ori, __global mwc64x_state_t* RandState){
 int n=get_global_id(0); 
 int x,y;
 double phi=0.0;
 double c=0.0;
 double ori=0.0;
 mwc64x_state_t rng = RandState[n];
 double ori_rand_term=random_uniform(&rng);
 RandState[n] = rng;

 y = n/YSTEP;
 x = (n%YSTEP)/XSTEP;

 
 if (x<=XSIZE*0.2) {ori=0.3;phi=0.7;}
 if (x>XSIZE*0.8 ) {ori=0.7;phi=0.7;}
 if (x>XSIZE*0.2 && x<XSIZE*0.8){phi=0.0;ori=ori_rand_term;}
 
 c=C_0;

 Phi[n]=phi;
 C[n]=c;
 double mod1=fmod(ori,1.0);
 Ori[n]=fmod(mod1+1.0,1.0);
 
}

__kernel void insert_rectangle(__global double* Phi, __global double* C,
 __global double* Ori, const double X0, const double Y0, const double A0,const double start_orientation, __global mwc64x_state_t* RandState){
 int n=get_global_id(0); 
 int x,y;
 double phi=0.0;
 double c=0.0;
 double ori=0.0;
 mwc64x_state_t rng = RandState[n];
 double ori_rand_term=random_uniform(&rng);
 RandState[n] = rng;

 y = n/YSTEP;
 x = (n%YSTEP)/XSTEP;
 
 
 if ((X0-A0)<x && x<=(X0+A0) && (Y0-A0)<y && y<=(Y0+A0)) 
 {
 	Phi[n]=0.8;
 	Ori[n]=start_orientation;
 	C[n]=C_0;
 }
 
}

__kernel void insert_orientation_defect_conf(__global double* Phi, __global double* C,  __global double* Ori, __global mwc64x_state_t* RandState,
 const double epsilon,const double dx0){
 int n=get_global_id(0); 
 int x,y;
 double phi=0.0;
 double c=0.0;
 double ori=0.0;
 mwc64x_state_t rng = RandState[n];
 double ori_rand_term=random_uniform(&rng);
 RandState[n] = rng;

 y = n/YSTEP;
 x = (n%YSTEP)/XSTEP;

 double theta2=0.25+epsilon/2.0;
 double theta1=0.75-epsilon/2.0;
 double theta3=0.5;
 

if (x<XSIZE*0.45)
{
	Phi[n]=(1.0+tanh((0.4*XSIZE-x)*5.0))*0.5;
	Ori[n]=theta1+(1.0-(1.0+tanh((0.4*XSIZE+5-x)*5.0))*0.5)*random_uniform(&rng);
}

if (x>XSIZE*0.55)
{
	Phi[n]=(1.0+tanh((x-0.6*XSIZE)*5.0))*0.5;
	Ori[n]=theta2+(1.0-(1.0+tanh((x-0.6*XSIZE+dx0)*5.0))*0.5)*random_uniform(&rng);
}


if (XSIZE*0.45<x && x<XSIZE*0.55)
{
 Phi[n]=0;
 Ori[n]=random_uniform(&rng);
}

 
 double rx2=(x-XSIZE*0.5)*(x-XSIZE*0.5);
 double ry2=(y-YSIZE*0.5)*(y-YSIZE*0.5);

 double r=sqrt(rx2+ry2);
 double ori_ref;
 double r_0=50.0;

 if (r<r_0){
  phi    =(1.0-tanh((r-r_0/2.0)*0.5))/2.0;
  ori_ref=(1.0-tanh((r-(5+r_0/2.0))*0.5))/2.0;
  c=C_0;
  ori=theta3+(1.0-ori_ref)*ori_rand_term;
  
 Phi[n]=phi;
 C[n]=c;
 double mod1=fmod(ori,1.0);
 Ori[n]=fmod(mod1+1.0,1.0);;
 }
 
}



__kernel void insert_orientation_defect_conf_stepfun(__global double* Phi, __global double* C,  __global double* Ori, const double epsilon){
 int n=get_global_id(0); 
 int x,y;
 double phi=0.0;
 double c=0.0;
 double ori=0.0;

 y = n/YSTEP;
 x = (n%YSTEP)/XSTEP;

 double theta2=0.25+epsilon/2.0;
 double theta1=0.75-epsilon/2.0;
 double theta3=0.5;
 

if (x<XSIZE*0.5)
{
  Phi[n]=0.9;
  Ori[n]=theta1;
}

if (x>=XSIZE*0.5)
{
  Phi[n]=0.9;
  Ori[n]=theta2;
}



 double rx2=(x-XSIZE*0.5)*(x-XSIZE*0.5);
 double ry2=(y-YSIZE*0.5)*(y-YSIZE*0.5);

 double r=sqrt(rx2+ry2);
 double ori_ref;
 double r_0=50.0;

 if (r<r_0){
  phi    =0.9;
  c=C_0;
  ori=theta3;
  
 Phi[n]=phi;
 C[n]=c;
 double mod1=fmod(ori,1.0);
 Ori[n]=fmod(mod1+1.0,1.0);
 }
 
}

__kernel void insert_orientation_defect_conf_stepfun_cross(__global double* Phi, __global double* C,  __global double* Ori, const double epsilon){
 int n=get_global_id(0); 
 int x,y;
 double phi=0.0;
 double c=0.0;
 double ori=0.0;

 y = n/YSTEP;
 x = (n%YSTEP)/XSTEP;

 double theta2=0.25+epsilon/2.0;
 double theta1=0.75-epsilon/2.0;
 double theta3=0.5;
 

if (x<y)
{
  Phi[n]=1.0;
  Ori[n]=theta1;
}

if (x>=y)
{
  Phi[n]=1.0;
  Ori[n]=theta2;
}



 double rx2=(x-XSIZE*0.5)*(x-XSIZE*0.5);
 double ry2=(y-YSIZE*0.5)*(y-YSIZE*0.5);

 double r=sqrt(rx2+ry2);
 double ori_ref;
 double r_0=50.0;

 if (r<r_0){
  phi    =1.0;
  c=C_0;
  ori=theta3;
  
 Phi[n]=phi;
 C[n]=c;
 double mod1=fmod(ori,1.0);
 Ori[n]=fmod(mod1+1.0,1.0);
 }
 
}