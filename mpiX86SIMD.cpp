#include<stdio.h>
#include"mpi.h"
#include <iostream>
#include <windows.h>
#include<stdio.h>
#include<time.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
using namespace std;
float** A;
float** B;
float** C;
float** D;
float** E;
float* recv_arr;
int n =1000;
const int N = 1000;
const int thread_count = 4;
float mat[N][N];
void init_mat(float test[][N])
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            test[i][j] = rand() / 100;
}
void print_mat(float mat[][N])
{
    if (N > 16)
        return;
    cout << endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            cout << mat[i][j] << " ";
        cout << endl;
    }
    cout << endl;
}
void init(int n)
{
    A = new float* [n];
    recv_arr = new float[n];
    for (int i = 0; i < n; i++)
        A[i] = new float[n];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
            A[i][j] = 0.0;
        A[i][i] = 1.0;
        for (int j = i + 1; j < n; j++)
            A[i][j] = rand();
    }
    for (int i = 1; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            A[i][j] += A[i - 1][j];
    }


}
void deepcopy()
{
    B = new float* [n];
    for (int i = 0; i < n; i++)
        B[i] = new float[n];
    C = new float* [n];
    for (int i = 0; i < n; i++)
        C[i] = new float[n];
    D = new float* [n];
    for (int i = 0; i < n; i++)
        D[i] = new float[n];
    E = new float* [n];
    for (int i = 0; i < n; i++)
        E[i] = new float[n];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            B[i][j] = A[i][j];
            C[i][j] = A[i][j];
            D[i][j] = A[i][j];
            E[i][j] = A[i][j];
        }
    }
}
void eliminate_opt(float mat[][N], int rank, int num_proc)
{
    __m128 t1, t2, t3;
    int block = N / num_proc;
    int remain = N % num_proc;
    int begin = rank * block;
    int end = rank != num_proc - 1 ? begin + block : begin + block + remain;
//#pragma omp parallel num_threads(thread_count)
    for (int k = 0; k < N; k++)
    {
        if (k >= begin && k < end)
        {
            float temp1[4] = { mat[k][k], mat[k][k], mat[k][k], mat[k][k] };
            t1 = _mm_loadu_ps(temp1);
            int j = k + 1;
//#pragma omp for schedule(guided, 20)
//#pragma omp for
            for (j=k+1; j < N - 3; j += 4)
            {
                t2 = _mm_loadu_ps(mat[k] + j);
                t3 = _mm_div_ps(t2, t1);
                _mm_storeu_ps(mat[k] + j, t3);
            }
//#pragma omp for schedule(guided, 20)
//#pragma omp for
            for (int m=j; m < N; m++)
            {
                mat[k][j] = mat[k][j] / mat[k][k];
                j++;
            }
            mat[k][k] = 1.0;
            for (int p = rank + 1; p < num_proc; p++)
                MPI_Send(&mat[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
        }
        else
        {
            int cur_p = k / block;
            if (cur_p < rank)
                MPI_Recv(&mat[k], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int i = begin; i < end && i < N; i++)
        {
            if (i >= k + 1)
            {
                float temp2[4] = { mat[i][k], mat[i][k], mat[i][k], mat[i][k] };
                t1 = _mm_loadu_ps(temp2);
                int j = k + 1;
//#pragma omp for schedule(guided, 20)
//#pragma omp for
                for (j=k+1; j <= N - 3; j += 4)
                {
                    t2 = _mm_loadu_ps(mat[i] + j);
                    t3 = _mm_loadu_ps(mat[k] + j);
                    t3 = _mm_mul_ps(t1, t3);
                    t2 = _mm_sub_ps(t2, t3);
                    _mm_storeu_ps(mat[i] + j, t2);
                }
//#pragma omp for schedule(guided, 20)
//#pragma omp for
                for (int m = j; m < N; m++)
                {
                    mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                    j++;
                }
                mat[i][k] = 0;
            }
        }
    }
}
void reset_mat(float** mat, float** test)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mat[i][j] = test[i][j];
}
int main(int argc, char* argv[])
{
    init(n);
    deepcopy();
    int rank, num_proc;
    int  namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Get_processor_name(processor_name, &namelen);
    long long head, tail, freq;
    
    int block = N / num_proc;
    int remain = N % num_proc;
    if (rank == 0)
    {
        init_mat(mat);
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Send(&mat[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Send(&mat[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        }
        eliminate_opt(mat, rank, num_proc);
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Recv(&mat[i * block + j], N, MPI_FLOAT, i, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Recv(&mat[i * block + j], N, MPI_FLOAT, i, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        //	cout <<"静态"<<(tail-head)*1000.0/freq<<"ms"<<endl;
        cout << "mpishijian          " << (tail - head) * 1000.0 / freq << "            ";
        //print_mat(mat);
    }
    else
    {
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Recv(&mat[rank * block + j], N, MPI_FLOAT, 0, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Recv(&mat[rank * block + j], N, MPI_FLOAT, 0, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        eliminate_opt(mat, rank, num_proc);
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Send(&mat[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Send(&mat[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;


}