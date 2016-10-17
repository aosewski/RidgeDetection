/**
 * @file evolve.hpp
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of
 * estimation of multidimensional random variable density function ridge
 * detection algorithm."
 * , which is supervised by prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and
 * Information Technology Warsaw University of Technology 2016
 * 
 * @note Slightly modified version of original Marek Rupniewski's
 * ridge detection algorithm. Changes are limited to making c++
 * function template version.
 * 
 */

#ifndef CPU_MRUP_EVOLVE_HPP
#define CPU_MRUP_EVOLVE_HPP

#include "util.hpp"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_statistics_float.h>
#include <gsl/gsl_machine.h>


/* Każdy wybraniec przesuwany jest do środka masy punktów sterujących
 * wpadających w komórkę wybrańca. Komórka wybrańca to przecięcie komórki
 * Voronoia wybrańca (dla diagramu Voronoia skonstruowanego dla wszystkich
 * wybrańców) i kuli o promieniu r.
 * Zaprzestanie ewolucji jeśli nie było w danej iteracji "numerycznie zauważalnej" zmiany któregokolwiek z wybrańców.
 */

void evolve(
    double *punkty,//< początek tablicy ze współrzędnymi (punkty "sterujące" ewolucją)
              //  każdy punkt reprezentowany jest przez dim współrzędnych,
    int num,  //< liczba punktów (każdy po dim współrzędnych)
    double r, //< parametr funkcji choose (różne znaczenie w zależności od implementacji).
    double *srodki,  //< początek tablicy z "wybrańcami" podlegającymi ewolucji
              //  (punkty po dim współrzędnych każdy)
    int liczba,//< liczba punktów podlegających ewolucji
    int dim, //< wymiar danych
    void *pozostale   //< wskaźnik do struktury zawierającej pozostałe argumenty funkcji (w razie potrzeby)
)
{
  double *p, *sum, *o, r2=r*r;
  int *licz_sum;
  int kontynuacja = 1;
  p         = new double[liczba * dim];
  sum       = new double[liczba * dim];
  licz_sum  = new int[liczba];
  o         = new double[liczba];

  assert(p        != NULL);
  assert(sum      != NULL);
  assert(licz_sum != NULL);
  assert(o        != NULL);


  while (kontynuacja) {
//    printf("kontynuacja\n");
    memset(sum     , 0, liczba * dim * sizeof(double));
    memset(licz_sum, 0, liczba * sizeof(int));
    double *punkt = punkty;
    for(int n = 0; n < num; n++, punkt += dim) {// po wszystkich punktach punkty
      memcpy(p, srodki, liczba * dim * sizeof(double)); // przepisanie środków
      memset(o, 0, liczba * sizeof(double));  // wyzerowanie tablicy odległości
      double *tp = p;
      for(int k=0; k<liczba; k++) {        // po wszystkich środkach
        for(int d=0; d<dim; d++, tp++) {
          *tp -= punkt[d];        // odjęcie współrzednych punktu
          o[k] += *tp * *tp;
        }
      }
      int ni = gsl_stats_min_index(o,1,liczba); // indeks najbliższego środka
//      printf("liczba = %d, punkt nr %3d, ni=%d\n",liczba, n, ni);
      if (o[ni] <= r2) {
        licz_sum[ni]++;
        for(int d=0; d<dim; d++) sum[ni*dim + d] += p[ni*dim + d];
//        printf("n=%4d, ni=%4d, p[ni*dim+0]=%f\n", n, ni, p[ni*dim]);
      }
    }
    kontynuacja = 0;
    for(int k = 0; k < liczba; k++) // wyliczenie średnich
      for(int d=0; d<dim; d++) {
        double sr = srodki[k*dim + d] - sum[k*dim +d] / licz_sum[k];
        if (fabs(sr -srodki[k*dim + d]) > 2* fabs(sr)*(licz_sum[k] * GSL_DBL_EPSILON))
//          kontynuacja = 1;
        {
          // printf("duża zmiana k=%5d, d=%2d, sr=%10.6f, srodki=%10.6f, licz_sum=%7d, 
          // roznica=%g, por=%g\n",k, d, sr, srodki[k*dim+d], licz_sum[k],
          //    fabs(sr -srodki[k*dim + d]),  fabs(sr)*(licz_sum[k] * GSL_DBL_EPSILON));
          kontynuacja = 1;
        }

        srodki[k*dim + d] = sr;
      }
  }

  delete[] o;
  delete[] licz_sum;
  delete[] sum;
  delete[] p;
}

void evolve(
    float *punkty,//< początek tablicy ze współrzędnymi (punkty "sterujące" ewolucją)
              //  każdy punkt reprezentowany jest przez dim współrzędnych,
    int num,  //< liczba punktów (każdy po dim współrzędnych)
    float r, //< parametr funkcji choose (różne znaczenie w zależności od implementacji).
    float *srodki,  //< początek tablicy z "wybrańcami" podlegającymi ewolucji
              //  (punkty po dim współrzędnych każdy)
    int liczba,//< liczba punktów podlegających ewolucji
    int dim, //< wymiar danych
    void *pozostale   //< wskaźnik do struktury zawierającej pozostałe argumenty funkcji (w razie potrzeby)
)
{
  float *p, *sum, *o, r2=r*r;
  int *licz_sum;
  int kontynuacja = 1;
  p         = new float[liczba * dim];
  sum       = new float[liczba * dim];
  licz_sum  = new int[liczba];
  o         = new float[liczba];

  assert(p        != NULL);
  assert(sum      != NULL);
  assert(licz_sum != NULL);
  assert(o        != NULL);


  while (kontynuacja) {
//    printf("kontynuacja\n");
    memset(sum     , 0, liczba * dim * sizeof(float));
    memset(licz_sum, 0, liczba * sizeof(int));
    float *punkt = punkty;
    for(int n = 0; n < num; n++, punkt += dim) {// po wszystkich punktach punkty
      memcpy(p, srodki, liczba * dim * sizeof(float)); // przepisanie środków
      memset(o, 0, liczba * sizeof(float));  // wyzerowanie tablicy odległości
      float *tp = p;
      for(int k=0; k<liczba; k++) {        // po wszystkich środkach
        for(int d=0; d<dim; d++, tp++) {
          *tp -= punkt[d];        // odjęcie współrzednych punktu
          o[k] += *tp * *tp;
        }
      }
      int ni = gsl_stats_float_min_index(o,1,liczba); // indeks najbliższego środka
//      printf("liczba = %d, punkt nr %3d, ni=%d\n",liczba, n, ni);
      if (o[ni] <= r2) {
        licz_sum[ni]++;
        for(int d=0; d<dim; d++) sum[ni*dim + d] += p[ni*dim + d];
//        printf("n=%4d, ni=%4d, p[ni*dim+0]=%f\n", n, ni, p[ni*dim]);
      }
    }
    kontynuacja = 0;
    for(int k = 0; k < liczba; k++) // wyliczenie średnich
      for(int d=0; d<dim; d++) {
        float sr = srodki[k*dim + d] - sum[k*dim +d] / licz_sum[k];
        if (fabs(sr -srodki[k*dim + d]) > 2* fabs(sr)*(licz_sum[k] * GSL_FLT_EPSILON))
//          kontynuacja = 1;
        {
          // printf("duża zmiana k=%5d, d=%2d, sr=%10.6f, srodki=%10.6f, licz_sum=%7d, roznica=%g,
          //  por=%g\n",k, d, sr, srodki[k*dim+d], licz_sum[k],
          //    fabs(sr -srodki[k*dim + d]),  fabs(sr)*(licz_sum[k] * GSL_FLT_EPSILON));
          kontynuacja = 1;
        }

        srodki[k*dim + d] = sr;
      }
  }

  delete[] o;
  delete[] licz_sum;
  delete[] sum;
  delete[] p;
}

#endif  // CPU_MRUP_EVOLVE_HPP
