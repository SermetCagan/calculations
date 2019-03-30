#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

const double M_P = 4.341e-9; // Planck Mass

// Number of e-foldings integral integrand
double integrand(double phi, double xi){
	return 0.5*phi*(pow(M_P,2) + 6*pow(phi,2)*pow(xi,2) + pow(phi,2)*xi)/(pow(M_P,4) - pow(phi,4)*pow(xi,2));
}

// End of inflation scalar field
double phi_f(double xi){
	return sqrt(pow(M_P,2)*(sqrt(48*pow(xi,2) + 16*xi + 1) - 1 - 4*xi)/(8*pow(xi,2) + 2*xi));
}

// Slow roll-over parameters
double epsilon(double phi, double xi){
	return 2.0*pow((pow(M_P,2) - pow(phi,2)*xi),2)/(pow(phi,2)*(pow(M_P,2) + 6*pow(phi,2)*pow(xi,2) + pow(phi,2)*xi));
}

double eta(double phi, double xi){
	return (2.0*pow(M_P,6) - 12.0*pow(M_P,4)*pow(phi,2)*xi - 72.0*pow(M_P,2)*pow(phi,4)*pow(xi,3) - 10.0*pow(M_P,2)*pow(phi,4)*pow(xi,2) + 24.0*pow(phi,6)*pow(xi,4) + 4.0*pow(phi,6)*pow(xi,3))/(pow(phi,2)*(pow(M_P,4) + 12.0*pow(M_P,2)*pow(phi,2)*pow(xi,2) + 2.0*pow(M_P,2)*pow(phi,2)*xi + 36.0*pow(phi,4)*pow(xi,4) + 12.0*pow(phi,4)*pow(xi,3) + pow(phi,4)*pow(xi,2)));
}

// Trapezoidal numerical integration
double trapezoidal(double (*f)(double,double), double xi, double a, double b, double n){
	double delta_x = (b-a)/n;
	double result = 0;
	
	for(int i = 1; i < n; i++){
		result += f(a + i*delta_x, xi);
	}
	result += 0.5*(f(a, xi) + f(b, xi));
	return result*delta_x;
}

double limit_finder(double (*f)(double,double), double xi, double a, double res, double dx){
	double temp = 0;
	int i = 0;
	while(fabs(temp - res) >= 1e-6){
		temp += 0.5*(f(a + i*dx, xi) + f(a + dx*(i+1), xi))*dx;
		i += 1;
	}
	return a + (i-1)*dx;
}

double ff(double x){
	return x;
}

void save_to_file(double *data1, double *data2, int size_of_data, char *file_name){
	FILE *fp;
	fp = fopen(file_name, "w+");
	for(int i = 0; i < size_of_data; i++){
		fprintf(fp, "%2.16e %2.16e\n", data1[i], data2[i]);
	}
	fclose(fp);
}

int main() {
	int number_of_xi_points = 1000;
	int number_of_integration_points = pow(2,8);
	int number_of_e_foldings = 60;

	double tolerance = 1e-1;
	double delta_phi = 1e-10;
	double xi[number_of_xi_points];

	double phi_60[number_of_xi_points];
	double phi_50[number_of_xi_points];

	double xi_ini = -0.002;
	double xi_fin = 0.005;
	double delta_xi = (xi_fin-xi_ini)/(number_of_xi_points);

	clock_t start, end;
	double cpu_time;
	
	int progress = 0;
	
	for(int i = 0; i < number_of_xi_points; i++){
		xi[i] = xi_ini + i*delta_xi;
	}

	for(int i = 0; i < number_of_xi_points; i++){
		start = clock();
		phi_60[i] = limit_finder(integrand, xi[i], phi_f(xi[i]), number_of_e_foldings, 1e-6);
		end = clock();
		printf("%f\n", ((double) end - start) / CLOCKS_PER_SEC);
		phi_50[i] = limit_finder(integrand, xi[i], phi_f(xi[i]), 50, 1e-6);
	}
	
//	for(int i = 0; i < number_of_xi_points; i++){
//		int iterator = 1;
//		double integration_result = 0;
//		while(fabs(number_of_e_foldings - integration_result) >= tolerance){
//			integration_result = trapezoidal(integrand, xi[i], phi_f(xi[i]), phi_f(xi[i]) + iterator*delta_phi, number_of_integration_points);
//			iterator++;
//		}
//		phi_60[i] = phi_f(xi[i]) + (iterator-1)*delta_phi;
//		progress++;
//		printf("\r%d",progress);	
//		fflush(stdout);
//	}


//	progress = 0;
//	for(int i = 0; i < number_of_xi_points; i++){
//		int iterator = 1;
//		double integration_result = 0;
//		int number_of_e_foldings = 50;
//		while(fabs(number_of_e_foldings - integration_result) >= tolerance){
//			integration_result = trapezoidal(integrand, xi[i], phi_f(xi[i]), phi_f(xi[i]) + iterator*delta_phi, number_of_integration_points);
//			iterator++;
//		}
//		phi_50[i] = phi_f(xi[i]) + (iterator-1)*delta_phi;
//		progress++;
//		printf("\r%d",progress);
//		fflush(stdout);
//	}
//
//	double ns60[number_of_xi_points];
//	double r60[number_of_xi_points];
//	double ns50[number_of_xi_points];
//	double r50[number_of_xi_points];
//
//	double eta60[number_of_xi_points];
//	double eps60[number_of_xi_points];
//	double eta50[number_of_xi_points];
//	double eps50[number_of_xi_points];
//	
//	for(int i = 0; i < number_of_xi_points; i++){
//		eta60[i] = eta(phi_60[i], xi[i]);
//		eps60[i] = epsilon(phi_60[i], xi[i]);
//		eta50[i] = eta(phi_50[i], xi[i]);
//		eps50[i] = epsilon(phi_50[i], xi[i]);
//	}
//
//	for(int i = 0; i < number_of_xi_points; i++){
//		ns60[i] = 1 + 2*eta60[i] - 6*eps60[i];
//		r60[i] = 16*eps60[i];
//		ns50[i] = 1 + 2*eta50[i] - 6*eps50[i];
//		r50[i] = 16*eps50[i];
//	}
//
//	
//	printf("\nFinished calculation...\nNow writing to file...\n");
//	save_to_file(ns60, r60, sizeof(phi_60)/sizeof(phi_60[0]), "/Users/sermetcagan/Desktop/60efolding_values.txt");
//	save_to_file(ns50, r50, sizeof(phi_60)/sizeof(phi_60[0]), "/Users/sermetcagan/Desktop/50efolding_values.txt");
//	return 0;
//
}
