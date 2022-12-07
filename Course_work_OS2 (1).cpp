#include <iostream>
#include <fstream>
#include <omp.h>
#include <stdlib.h>     // srand, rand
#include <time.h> 
#include "LinalGenerator.h" // generator function
// set data type
#define type double

double** make_matrix(int size)
{
	std::vector<std::vector<type>> generated = generateGoodConditionedMatrix<type>(size);
	type** matrix = new type * [size];
	for (int i = 0; i < size; ++i)
		matrix[i] = new type[size];
	for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j)
			matrix[i][j] = generated[i][j];
	return matrix;
}

type* make_random(int size)
// return random fill vector
{
	type* vector = new type[size];
	for (int i = 0; i < size; ++i)
		vector[i] = type(rand() % size) + 1;
	return vector;
}

type* make_zeros(int size)
// return zero vector
{
	type* vector = new type[size];
	for (int i = 0; i < size; ++i)
		vector[i] = 0;
	return vector;
}

bool check(type* a, type* b, int size, type eps)
// checking result
{
	for (int i = 0; i < size; ++i)
		if (abs(a[i] - b[i]) > eps)
			return false;
	return true;
}

type* matmul(type** matr, type* v, int size)
// matrix by vector multiplication
{
	type* result = new type[size];
	type temp = 0;
#pragma omp parallel for private(temp)
	for (int i = 0; i < size; ++i)
	{
		temp = 0;
		for (int j = 0; j < size; ++j)
			temp += v[j] * matr[i][j];
		result[i] = temp;
	}
	return result;
}

type residual, sigma; // used to reduction

void recompute_accuracy_non_parralel(type** A, type* phi, type* b, int size)
// Non parralel method
// find accuracy using parallel
// accuracy metric is "The Frobenius norm"
{
	type temp;
	residual = 0;
	for (int i = 0; i < size; ++i)
	{
		temp = -b[i];
		for (int j = 0; j < size; ++j)
			temp += A[i][j] * phi[j];
		residual += temp * temp;
	}
	residual = sqrt(residual);
}

type* low_sor_non_parralel(type** A, type* b, int size,
	type omega, type* initial, type convergence_criteria, double& time)
	// Non parralel method
	// function to find solution for Ax=b using lower SOR method
	// A - (size x size) matrix
	// vector b - right side of the equation
	// omega - parameter
	// initial - initial vector for x
	// residual - accuracy
{
	type* phi = new type[size];
	for (int i = 0; i < size; ++i) phi[i] = initial[i];
	double start = omp_get_wtime();		 // set start time
	recompute_accuracy_non_parralel(A, phi, b, size); // find accuracy
	while (residual > convergence_criteria)
	{
		for (int i = 0; i < size; ++i)
		{
			sigma = - (A[i][i] * phi[i]);
			for (int j = 0; j < size; ++j)
			{
				sigma += A[i][j] * phi[j];
			}
			phi[i] = (1 - omega) * phi[i] + (omega / A[i][i]) * (b[i] - sigma);
		}
		recompute_accuracy_non_parralel(A, phi, b, size); // find accuracy
	}
	double end = omp_get_wtime(); // set end time
	time = end - start;			  // compute time

	return phi; // return result (x)
}

void recompute_accuracy(type** A, type* phi, type* b, int size, int number_threads)
// find accuracy using parallel
// accuracy metric is "The Frobenius norm"
{
	type temp;
	residual = 0;
	omp_set_num_threads(number_threads);
#pragma omp parallel for private(temp) shared(residual) schedule(dynamic)
	for (int i = 0; i < size; ++i)
	{
		temp = -b[i];
		for (int j = 0; j < size; ++j)
			temp += A[i][j] * phi[j];

#pragma	omp critical
		residual += temp * temp;
	}
	residual = sqrt(residual);
}

type* low_sor(type** A, type* b, int size,
	type omega, type* initial, type convergence_criteria, int number_threads, double& time)
	// function to find solution for Ax=b using lower SOR method
	// A - (size x size) matrix
	// vector b - right side of the equation
	// omega - parameter
	// initial - initial vector for x
	// residual - accuracy
{
	type* phi = new type[size];
	for (int i = 0; i < size; ++i) phi[i] = initial[i];
	double start = omp_get_wtime();		 // set start time
	recompute_accuracy(A, phi, b, size, number_threads); // find accuracy
	while (residual > convergence_criteria)
	{
		for (int i = 0; i < size; ++i)
		{
			sigma = -(A[i][i] * phi[i]);
			for (int j = 0; j < size; ++j)
			{
				sigma += A[i][j] * phi[j];
			}

			phi[i] = (1 - omega) * phi[i] + (omega / A[i][i]) * (b[i] - sigma);
		}
		recompute_accuracy(A, phi, b, size, number_threads); // find accuracy
	}
	double end = omp_get_wtime(); // set end time
	time = end - start;			  // compute time

	return phi; // return result (x)
}


// Описание работы программы
// Сначало генерируем матрицу А
// Затем генерирую слуйным образом вектор х
// Умножаю Ax и получаю вектор b
// для A и b нахожу х методом нижней релаксации и сравниваю результат функцией check
void test()
{

	// ------------ Подготовительный блок -------------
	srand(time(NULL));
	// Set parametres
	int size = 1000;
	type omega = 0.9;
	type convergence_criteria = 0.001;
	// Generate matrix
	type** A = make_matrix(size);
	// Generate matrix x
	type* x = make_random(size);
	// Set initial vector
	type* initial = make_zeros(size);

	// file pointer 
	std::fstream fout;

	// opens an existing csv file or creates a new file. 
	fout.open("reportcard.csv", std::ios::out | std::ios::app);
	// Table Header
	fout << "Shape, OneWay, 1 thread,2 threads,4 threads,8 threads\n";

	double time = 0; // time check variable

	// ------------- Вычислительный блок -------------
	// Make Table
	for (int i = 50; i <= size; i += 50)
	{
		// Find b = Ax
		// Перевычисляем каждый раз вектор b, так как меняем размерность
		type* b = matmul(A, x, i);
		fout << i;													// Put Shape
		type* result_np = low_sor_non_parralel(A, b, i,			// matrix A and vector b
			omega, initial, convergence_criteria, time);			// computetion parameters
		if (!check(result_np, x, i, convergence_criteria))			// check results
		{
			std::cout << "Different results!!!\n";
			for (int k = 0; k < i; ++k)
				std::cout << result_np[k] << " " << x[k] << std::endl;
			return;
		}
		fout << ", " << time;		// Put OneWay time
		delete[] result_np;
		for (int j = 1; j < 9; j*=2)
		{
			type* result = low_sor(A, b, i,				// matrix A and vector b
				omega, initial, convergence_criteria,		// computetion parameters
				j, time);									// checkin parameters
			if (!check(result, x, i, convergence_criteria)) // check results
			{
				std::cout << "Different results!!!\n";
				for (int k = 0; k < i; ++k)
					std::cout << result[k] << " " << x[k] << std::endl;
				return;
			}
			fout << ", " << time;							// Put parralel time
			delete[] result;
		}

		fout << "\n";		// end table string
		std::cout << i << std::endl; // Выводим счетчик для того чтобы наблюдать за процессом тестирования

		delete[] b;
	}

	// Clear memory
	for (int i = 0; i < size; ++i) delete[] A[i];
	delete[] A;
	delete[] x;
	delete[] initial;
}


void simple_test()
{
	int size = 1000;
	type omega = 0.5;
	type convergence_criteria = 0.000001;
	// Generate matrix
	type** A = make_matrix(size);
	// Generate matrix x
	type* x = make_random(size);
	// Find b = Ax
	type* b = matmul(A, x, size);
	std::cout << "\n";
	// Set initial vector
	type* initial = make_zeros(size);
	double time;
	// find solution
	type* result = low_sor(A, b, size, omega, initial, convergence_criteria, 4, time);
	// print solution
	std::cout << check(x, result, size, convergence_criteria) << std::endl;
	for (int k = 0; k < size; ++k)
		std::cout << result[k] << " " << x[k] << std::endl;

	for (int i = 0; i < size; ++i) delete[] A[i];
	delete[] A;
	delete[] b;
	delete[] x;
	delete[] initial;
	delete[] result;
}

void omega_test()
{

	// ------------ Подготовительный блок -------------
	srand(time(NULL));
	// Set parametres
	int size = 1000;
	type convergence_criteria = 0.01;
	// Generate matrix
	type** A = make_matrix(size);
	// Generate matrix x
	type* x = make_random(size);
	// Set initial vector
	type* initial = make_zeros(size);
	// Find b = Ax
	type* b = matmul(A, x, size);
	// file pointer 
	std::fstream fout;

	// opens an existing csv file or creates a new file. 
	fout.open("reportcard_omega.csv", std::ios::out | std::ios::app);
	// Table Header
	fout << "Omega, OneWay, 1 thread,2 threads,3 threads,4 threads\n";

	double time = 0; // time check variable

	// ------------- Вычислительный блок -------------
	// Make Table
	for (type omega = 0.1; omega <= 1; omega += 0.1)
	{
		fout << omega;													// Put Omega
		type* result_np = low_sor_non_parralel(A, b, size,				// matrix A and vector b
			omega, initial, convergence_criteria, time);				// computetion parameters
		if (!check(result_np, x, size, convergence_criteria))			// check results
		{
			std::cout << "Different results!!!\n";
			for (int k = 0; k < size; ++k)
				std::cout << result_np[k] << " " << x[k] << std::endl;
			return;
		}
		delete[] result_np;
		fout << ", " << time;										// Put OneWay time
		for (int j = 1; j < 5; ++j)
		{
			type* result = low_sor(A, b, size,				// matrix A and vector b
				omega, initial, convergence_criteria,		// computetion parameters
				j, time);									// checkin parameters
			if (!check(result, x, size, convergence_criteria)) // check results
			{
				std::cout << "Different results!!!\n";
				for (int k = 0; k < size; ++k)
					std::cout << result[k] << " " << x[k] << std::endl;
				return;
			}
			fout << ", " << time;							// Put parralel time
			delete[] result;
		}

		fout << "\n";		// end table string
		std::cout << omega << std::endl; // Выводим omega для того чтобы наблюдать за процессом тестирования

	}

	// Clear memory
	for (int i = 0; i < size; ++i) delete[] A[i];
	delete[] A;
	delete[] x;
	delete[] initial;
	delete[] b;
}

int main()
{
	test();
}

