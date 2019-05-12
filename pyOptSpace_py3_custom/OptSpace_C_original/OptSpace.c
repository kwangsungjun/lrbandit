#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "svdlib.h"
#include "OptSpace.h"


double recongetoptT(SMat M,ReconVar x, double **W, double **Z,int Verbosity );
void recongetoptSiter(SMat M, ReconVar x,SMat XSYt, SMat D, double **XS);
void recongetoptSappx1(SMat M,ReconVar x);
void reconSVD(SMat M,ReconVar x);
void recongradFt(ReconVar x, double **XS, double **YSt,SMat D, double **W, double **Z);

double reconFt(SMat M,double **Xt, double **Yt, double **St, int r,double **XtSt, SMat XSYt, SMat D, int flag);

void recongeoD(double **Xt, double **X,SVDRec LRx, double t);


void setrn(int iseed[] ) ;

double fpen( double x, double m ,int i,int j);
double dfpen( double x, double m ,int i, int j);



ReconVar OptSpace(
SMat M,
int r,
int niter,
double tol,
int Verbosity,
char *outfile
)
{
	
	int i,j,k;
	int iter;
	double t;
	double dist;

/* * * * * * RND * * * * *  */
	int iseed[4];
	srand(time(NULL));
	for(i=0; i<4; ++i)
		iseed[i] = rand() % 4096;
	setrn(iseed);
/* * * * * * * * * * * * * */

	if(niter == 0)
		niter = 1000 ;
	
	double relfac0 = 0 ;
	
	for(i=0; i<M->vals; ++i)
		relfac0 += M->value[i]*M->value[i] ;
	
	if( tol == 0 )
	{
		tol = 1e-4 ;
	}	


		


	if(Verbosity > 0 )
	{
		printf("No. of rows             : %d\n",M->rows);
		printf("No. of cols             : %d\n",M->cols);
		printf("No. of revealed entries : %d\n",M->vals);
		printf("Using rank              : %d\n",r);
		printf("\n");
	}


	int n, m, E ;
	n = M->rows;
	m = M->cols;
	E = M->vals;

/* * * * * Variables * * * * */
ReconVar x ;

double** W;
double** Z;

SMat D;
SMat XSYt;
double** XS;
double** YSt;


/* * * * * * * Allocate Memory * * * * * * * */
	W = calloc(n,sizeof(double*)) ;
	Z = calloc(m,sizeof(double*)) ;


	for(i=0; i<n; ++i)
		W[i] = calloc(r,sizeof(double)) ;
	
	for(i=0; i<m; ++i)
		Z[i] = calloc(r,sizeof(double)) ;
	
	x = (ReconVar) calloc(1,sizeof(struct reconvar));
	x->rows = M->rows;
	x->cols = M->cols;
	x->rank = r;
	
	x->X = calloc(n,sizeof(double*)) ;
	x->Y = calloc(m,sizeof(double*)) ;
	x->S = calloc(r,sizeof(double*)) ;
	for(i=0; i<n; ++i)
		x->X[i] = calloc(r,sizeof(double)) ;
	
	for(i=0; i<m; ++i)
		x->Y[i] = calloc(r,sizeof(double)) ;
	
	for(i=0; i<r; ++i)
		x->S[i] = calloc(r,sizeof(double)) ;



	D = (SMat) calloc(1,sizeof(struct smat));
	D->rows = M->rows;
	D->cols = M-> cols;
	D->vals = M->vals;
	D->pointr = M->pointr;
	D->rowind = M->rowind;
	D->value = calloc(E,sizeof(double));
	
	XSYt = (SMat) calloc(1,sizeof(struct smat));
	XSYt->rows = M->rows;
	XSYt->cols = M-> cols;
	XSYt->vals = M->vals;
	XSYt->pointr = M->pointr;
	XSYt->rowind = M->rowind;
	XSYt->value = calloc(E,sizeof(double));

	XS = calloc(n,sizeof(double*));
	for(i=0; i<n; ++i)
		XS[i] = calloc(r,sizeof(double)) ;
	
	YSt = calloc(m,sizeof(double*));
	for(i=0; i<m; ++i)
		YSt[i] = calloc(r,sizeof(double)) ;
/* * * * End of Allocate Memory * * * * */


/* * * * Initial Guess * * * */	
	if( Verbosity > 1  )
	 	SVDVerbosity = 1;
	else
		SVDVerbosity = 0;

/* * * * SVD * * * */
	reconSVD(M,x) ;

	double rescal_param = sqrt(x->rank*M->vals/relfac0);
	for(i=0; i<M->vals; ++i)
		M->value[i] = M->value[i]*rescal_param ;
	relfac0 = relfac0 * rescal_param * rescal_param ;

	tol = tol*tol*relfac0 ;

/* * * * Min. f(X,Y,S) * * * */
	recongetoptSappx1(M,x);
    recongetoptSiter(M,x,XSYt,D,XS);
	dist = reconFt(M,x->X,x->Y,x->S,x->rank,XS,XSYt,D,1) ;
	
	int fverbosity = 1;
	FILE *fout;	
	if( strcmp(outfile,"") == 0 )
		fverbosity = 0;
		
	if( fverbosity == 1)	
	{
		fout = fopen(outfile,"a");
		fprintf(fout,"#Iteration\t RMSE\t               Rel. Err.\n" );
		fprintf(fout,"%d\t     %e\t     %e\n",0,sqrt(dist/((double)E))/rescal_param,sqrt(dist/relfac0) );
	}	

	SVDVerbosity = 0;

	if( Verbosity == 1)
	{	
		printf("\n");
		printf("#Iteration\t RMSE\t               Rel. Err.\n" );
		printf("%d\t     %e\t     %e\n",0,sqrt(dist/((double)E))/rescal_param,sqrt(dist/relfac0) );
	}	
	else if( Verbosity >= 2 )
	{
		printf("\n");
		printf("Iteration %d (SVD)\n",0);
		printf("RMSE      : %e\n",sqrt(dist/((double)E))/rescal_param );
		printf("Rel. Err. : %e\n",sqrt(dist/relfac0)); 
		printf("\n");
	}

	for(iter = 0; iter < niter; ++iter)
	{

/* Compute the Gradient */	
		recongradFt(x,XS,YSt,D,W,Z);
		
		if( Verbosity >=2 )
		{
			printf("\nIteration %d\n",iter+1);
		}

/* Backtracking Line Search */
		t = recongetoptT(M,x,W,Z,Verbosity) ;
		
/* Min. f(X,Y,S) */    	
		recongetoptSiter(M,x,XSYt,D,XS);

/* Compute F(X,Y) */
		dist = reconFt(M,x->X,x->Y,x->S,x->rank,XS,XSYt,D,1) ;
		
		if( Verbosity >=2 )
		{
			printf("RMSE      : %e\n",sqrt(dist/((double)E)) /rescal_param );
			printf("Rel. Err. : %e\n",sqrt(dist/relfac0) ); 
			printf("\n");
		}
	
		if( Verbosity == 1)
			printf("%d\t     %e\t     %e\n",iter+1,sqrt(dist/((double)E))/rescal_param,sqrt(dist/relfac0) );
		
	if( fverbosity == 1 )
		fprintf(fout,"%d\t     %e\t     %e\n",iter+1,sqrt(dist/((double)E))/rescal_param,sqrt(dist/relfac0) );
		
		if (dist<tol) break;
	}

	if( fverbosity == 1 )
		fclose(fout) ;
	
/* * * Rescale it back!! * * */

	for(i = 0; i<x->rank; ++i)
		for(j=0; j<x->rank; ++j)
			x->S[i][j] = x->S[i][j] / rescal_param ;



/* * * Free Memory * * */
	for(i=0; i<n; ++i)
		free(W[i]);
	free(W);

	for(i=0; i<m; ++i)
		free(Z[i]);
	free(Z);

	for(i=0; i<n; ++i)
		free(XS[i]);
	free(XS);

	for(i=0; i<m; ++i)
		free(YSt[i]);
	free(YSt);


	
	free(D->value) ;
    free(XSYt->value);
	free(D);
	free(XSYt);

	return(x) ;
}






void recongetoptSiter(SMat M, ReconVar x,SMat XSYt, SMat D, double **XS)
{

/* * * * * * * * Exact method * * * * * * * */
/* Get Sopt iteratively as the sol. of (1-lam)X'P_E(XSY'-M)Y=0 */
	int r = x->rank;
	
	int n, m, E ;
    n = M->rows;
    m = M->cols;
    E = M->vals;

	int    iter,iter1,i,j,c;
	double t=1e-5;
	double dist0,dist1,dGrad,b,t0,F0;
	double normXdSYt,normdS,temp1,prod_S_dS,prod_M_XdSY,prod_XSY_XdSY;;
	double **grad;
	grad = callocArray(r,r) ;
	double **temp ;
	temp = callocArray(n,r) ;
	double tol1 = 1e-2;
	double dist_old ;
	
	// Do steepest descent method
    dist0 = reconFt(M,x->X,x->Y,x->S,x->rank, XS, XSYt,D, 1) ;
	dist_old = dist0*5 ;
	for (iter=0;iter<50;iter++) 
	{

		// Compute gradF_S
		sparsemul(D,x->Y,n,m,r,temp,0) ;
		matrixmul(x->X,temp,r,n,r,grad,2,1.0,0.0);
		dGrad=0;
		for (i=0;i<r;i++)
			for (j=0;j<r;j++)
			{
//				dS[i][j] = -1*grad[i][j];
				dGrad += grad[i][j]*grad[i][j];
			}

		// Break if ||gradF_S||<1e-15 
		if (dGrad/r/r<1e-15) {break;}
		

// 1. find optimal t* = -{...}/{(1-lam)||Pe(XdST')||^2 + mn*lam*||dS||^2}
        normdS = dGrad; 
		prod_S_dS = 0;
		for(i=0; i<r; ++i)
			for(j=0; j<r; ++j) 
	                      prod_S_dS += -x->S[i][j]*grad[i][j]; 
		normXdSYt=0.;
		prod_XSY_XdSY=0.;
		prod_M_XdSY=0.;
		matrixmul(x->X,grad,n,r,r,XS,0, -1.0, 0.0) ;
		for(i=0,c=0; i<E; ++i)
		{
			while(M->pointr[c+1] <= i) c++;
			temp1 = vectmul(XS[M->rowind[i]],x->Y[c],r) ; 
			normXdSYt 	+= temp1*temp1;
			prod_XSY_XdSY 	+= XSYt->value[i]*temp1;
			prod_M_XdSY 	+= M->value[i]*temp1;
		}
		
		t0 = -1*( prod_XSY_XdSY-prod_M_XdSY )/(double)(normXdSYt);

		 for (i=0;i<r;i++)
		      for (j=0;j<r;j++) 
			x->S[i][j] += -t0*grad[i][j];
		dist_old = dist0;  
    	dist0 = reconFt(M,x->X,x->Y,x->S,x->rank, XS, XSYt,D, 1) ;
		if( dist_old - dist0 <= dist0*tol1){break;}
		
	}

	freeArray(grad,r) ;
	freeArray(temp,n) ;
}

/********************************************************************************************/

void recongetoptSappx1(SMat M,ReconVar x)
{
/* * * * * * * * Approx. Method * * * * * * * */
/* Sopt = X'P_E(M)Y/((1-lam+lam*m*n/E)*E) 		  */
/* error terms O(sqrt(\eps*r)				  */
	int r = x->rank;
	
	int n, m, E ;
    n = M->rows;
    m = M->cols;
    E = M->vals;
	double **temp ;
	double alpha;
	alpha =  ((double)E) ;
	alpha = 1/alpha ;

	temp = callocArray(n,r) ;
	
	sparsemul(M,x->Y,n,m,r,temp,0) ;
	matrixmul(x->X,temp,r,n,r,x->S,2,alpha,0.0);
	

	freeArray(temp,n) ;
}

/********************************************************************************************/


double recongetoptT(SMat M,ReconVar x, double **W, double **Z,int Verbosity )
{

	int n, m, E, r ;
    n = M->rows;
    m = M->cols;
    E = M->vals;
	r = x->rank;

	int i,j,iter;
	double **Xt, **Yt;
	DMat W1, Z1;
	SMat WS, ZS;
	SVDRec LRx, LRy;

	Xt = calloc(n,sizeof(double*));
	for (i=0; i<n; ++i)
		Xt[i] = calloc(r,sizeof(double));
	
	Yt = calloc(m,sizeof(double*));
	for (i=0; i<m; ++i)
		Yt[i] = calloc(r,sizeof(double));
	

	if( r > 1 )
	{

	W1 = (DMat) malloc(sizeof(struct dmat));
	W1->rows = n;
	W1->cols = r;
	W1->value = W ;
	
	Z1 = (DMat) malloc(sizeof(struct dmat));
	Z1->rows = m;
	Z1->cols = r;
	Z1->value = Z ;

	WS = svdConvertDtoS(W1);
	ZS = svdConvertDtoS(Z1);

	
	LRx = svdLAS2A(WS,r);
	LRy = svdLAS2A(ZS,r);

	
	svdFreeSMat(WS);
	svdFreeSMat(ZS);


	free(W1);
	free(Z1);
	
	}
	else
	{
		LRx = svdNewSVDRec();
		LRx->d = 1;
		LRx->Ut = svdNewDMat(1,n);
		
		double sing;
		

		sing = normF2(W,n,r);
		sing = sqrt(sing);
		for( i=0; i<n; ++i)
			LRx->Ut->value[0][i] = W[i][0] /sing;
		
		LRx->S = malloc(1*sizeof(double));	
		LRx->S[0] = sing ;
		LRx->Vt = svdNewDMat(1,1);
		LRx->Vt->value[0][0] = 1;
		
		
		
		LRy = svdNewSVDRec();
		LRy->d = 1;
		LRy->Ut = svdNewDMat(1,m);
		
		sing = normF2(Z,m,r);
		sing = sqrt(sing);
		for( i=0; i<m; ++i)
			LRy->Ut->value[0][i] = Z[i][0]/sing ;
		
		LRy->S = malloc(1*sizeof(double));	
		LRy->S[0] = sing ;
		LRy->Vt = svdNewDMat(1,1);
		LRy->Vt->value[0][0] = 1;

	}



	double dist;
	double **dummy0;
	SMat dummy1; 
	dummy0 = callocArray(n,r) ;

	dist = reconFt(M,x->X,x->Y,x->S,x->rank,  dummy0, dummy1, dummy1, 0 ) ;
	
	
	double norm2WZ = normF2(W,n,r) + normF2(Z,m,r)  ;

	double alpha = 1e-3 ;
	double distt;
	

	if( Verbosity >= 2 )
	{
		printf("Norm of Grad : %e\n",norm2WZ );
		printf("Backtracking Line Search : \n");
		printf("\t   Time Step \t      F(xt) \n");
	}



	for(iter = 0; iter<20; ++iter)
	{	
		recongeoD(Xt,x->X,LRx,-alpha);
		recongeoD(Yt,x->Y,LRy,-alpha);
	

		distt = reconFt(M,Xt,Yt,x->S,x->rank,  dummy0, dummy1, dummy1, 0 ) ;
		
		if( Verbosity >= 2)
			printf("\t %e     %e\n",alpha,distt);


		if( distt - dist <= -.5*alpha*norm2WZ )
		{	
			break;
		}
		
		alpha = alpha/2 ;
	}	
	
	if( Verbosity >= 2 )
	{
		printf("\nF(x_0) : %e\n",dist);
		printf("F(x_t*): %e\n",distt);
	}	

	for(i=0; i<n; ++i)
		for(j=0; j<r; ++j) {
			x->X[i][j] = Xt[i][j] ;
                }
	
	for(i=0; i<m; ++i)
		for(j=0; j<r; ++j) {
			x->Y[i][j] = Yt[i][j] ;
                }



	for (i=0; i<n; ++i)
		free(Xt[i]);
	free(Xt);

	for (i=0; i<m; ++i)
		free(Yt[i]);
	free(Yt);	

	freeArray(dummy0,n) ;
	svdFreeSVDRec(LRx);
	svdFreeSVDRec(LRy);

	return( (double) alpha) ;

}






/***************************************************/
/******************* Gradient **********************/

void recongradFt(ReconVar x, double **XS, double **YSt,SMat D, double **W, double **Z)
{
	int i, j, k;
	int c;
	
	int n, m,r  ;
	n = x->rows;
	m = x->cols;
	r = x->rank;

	double **temp ;
	temp = calloc(r,sizeof(double*));
	for(i=0; i<r; ++i)
		temp[i] = calloc(r,sizeof(double)) ;

	matrixmul(x->Y,x->S,m,r,r,YSt,1, 1.0, 0.0) ;

	sparsemul(D,YSt,n,m,r,W,0);
	sparsemul(D,XS,m,n,r,Z,2);

	matrixmul(x->X,W,r,n,r,temp,2, 1.0/n, 0.0) ;
	matrixmul(x->X,temp,n,r,r,W,0, -1.0, 1.0) ;
	
	matrixmul(x->Y,Z,r,m,r,temp,2, 1.0/m, 0.0) ;
	matrixmul(x->Y,temp,m,r,r,Z,0, -1.0, 1.0) ;

	for(i=0; i<r; ++i)
		free(temp[i]) ;
	free(temp);	
		

}





void reconSVD(SMat M,ReconVar x)
{
	int i,k;
	int n, m,E,r;
	n = M->rows;
	m = M->cols;
	E = M->vals;
	r = x->rank;

	double sqn = sqrt( (double) n);
	double sqm = sqrt( (double) m);
	SVDRec Out ;
	double end[2] = {-1.0e-30, 1.0e-30};	


/***** SVD Function goes here!! *****/
	Out = svdLAS2B(M,r);
	
	r = Out->d ;

	


	for(i=0; i<n; ++i)
		for(k=0; k<r; ++k)
		{
			x->X[i][k] = sqn * Out->Ut->value[k][i] ;
		}	
	
	for(i=0; i<m; ++i)
		for(k=0; k<r; ++k)
		{
			x->Y[i][k] = sqm * Out->Vt->value[k][i] ;
		}	


	for(k=0; k<r; ++k)
	{
		x->S[k][k] = (sqn*sqm)*Out->S[k] /((double)E);
	}	
	

	svdFreeSVDRec(Out) ;
}

double reconFt(SMat M,double **Xt, double **Yt, double **St,int r, double **XtSt, SMat XSYt, SMat D, int flag)
{
	
	int i,c ;
	double sum = 0;
	double temp;
	int n, m, E ;
	n = M->rows;
	m = M->cols;
	E = M->vals;

	if( flag == 1 )
	{
	matrixmul(Xt,St,n,r,r,XtSt,0, 1.0, 0.0) ;
		for(i=0,c=0; i<M->vals; ++i)
		{
			while(M->pointr[c+1] <= i) c++;
			temp = vectmul(XtSt[M->rowind[i]],Yt[c],r) ; 	// temp = (XSY')_{row(i),c}
			D->value[i] =  dfpen(temp,M->value[i],M->rowind[i],c); 
			XSYt->value[i] = temp;
			sum += fpen(temp, M->value[i],M->rowind[i],c);
		}
	}
	else
	{
		matrixmul(Xt,St,n,r,r,XtSt,0, 1.0, 0.0) ;
		for(i=0,c=0; i<M->vals; ++i)
		{
			while(M->pointr[c+1] <= i) c++;
			temp = vectmul(XtSt[M->rowind[i]],Yt[c],r) ; 	// temp = (XSY')_{row(i),c}
			sum += fpen(temp, M->value[i],M->rowind[i],c);
		}
	
	}



	return(sum) ;
}


/*****************************************************************************************************/



double** callocArray(int n1, int n2)
{
	int i;
	double **A;
	A = calloc(n1,sizeof(double*));
	for(i=0; i<n1; ++i)
		A[i] = calloc(n2,sizeof(double));
	return(A) ;	
}

double** mallocArray(int n1, int n2)
{
	int i;
	double **A;
	A = malloc(n1*sizeof(double*));
	for(i=0; i<n1; ++i)
		A[i] = malloc(n2*sizeof(double));
	return(A) ;	
}

void freeArray(double **A, int n1)
{
	int i;
	for(i=0; i<n1; ++i)
		free(A[i]);
	free(A);	
}


void freeReconVar(ReconVar x)
{
	int i;
	
	int n, m, r ;
	n = x->rows;
	m = x->cols;
	r = x->rank;
	



	for(i=0; i<n; ++i)
		free(x->X[i]);
	
	for(i=0; i<m; ++i)
		free(x->Y[i]);
	
	for(i=0; i<r; ++i)
		free(x->S[i]);
	


	free(x->X);
	free(x->Y);
	free(x->S);
	free(x);
}

/***************************************************/

double fpen( double x, double m ,int i, int j)
{
  return (    (x-m)*(x-m) );
} 

/***************************************************/

double dfpen( double x, double m,int i, int j )
{
  return (2.*(x-m) );
}

void recongeoD(double **Xt, double **X, SVDRec LRx, double t)
{
	int n1,i,j,r1,m1;
	double **T, **T1,**T2 ;
	n1 = LRx->Ut->cols;
	r1 = LRx->d;
	m1 = LRx->Vt->cols;

	T = callocArray(r1,r1);
	T1 = callocArray(r1,m1);
	T2 = callocArray(m1,m1);


	for(i=0; i<r1; ++i)
		T[i][i] = cos(t*(LRx->S[i])) - 1 ;


	matrixmul(T,LRx->Vt->value,r1,r1,m1,T1,0,1.0,0.0);
	
	matrixmul(LRx->Vt->value,T1,m1,r1,m1,T2,2,1.0,0.0);


	matrixmul(X,T2,n1,m1,m1,Xt,0,1.0,0.0);

	for(i=0; i<r1; ++i)
		for(j=0; j<r1; ++j)
			T[i][i] = 0;
			
	for(i=0; i<r1; ++i)
		T[i][i] = sin(t*(LRx->S[i])) ;
	

	matrixmul(T,LRx->Vt->value,r1,r1,m1,T1,0,1.0,0.0);

	matrixmul(LRx->Ut->value,T1,n1,r1,m1,Xt,2,sqrt(n1),1.0);


	for(i=0; i<n1; ++i)
		for(j=0; j<m1; ++j)
			Xt[i][j] += X[i][j] ;


	for(i=0; i<r1; ++i)
		free(T[i]);
	free(T);

	for(i=0; i<r1; ++i)
		free(T1[i]);
	free(T1);

	for(i=0; i<m1; ++i)
		free(T2[i]);
	free(T2);	



}

