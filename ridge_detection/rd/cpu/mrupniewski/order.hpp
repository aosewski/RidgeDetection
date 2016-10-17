/**
 * @file order.hpp
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

#ifndef CPU_MRUP_ORDER_HPP
#define CPU_MRUP_ORDER_HPP

#include "util.hpp"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

#include "sglib.h"

template <typename T>
struct listaPunktow {
  T *p;
  struct listaPunktow<T> *next_point;// kolejny punkt w ciągu
  struct listaPunktow<T> *next_list; // kolejny ciąg punktów
};


// funkcja łącząca ciągi listy lista w jeden
template <typename T>
listaPunktow<T> *join(listaPunktow<T> *lista) {
  int liczbaSkladowych;
  SGLIB_LIST_LEN(listaPunktow<T>, lista, next_list, liczbaSkladowych);

  while( liczbaSkladowych > 1) {
    listaPunktow<T> *pkoncowy, *pk, *dolaczanyCiag = NULL;
    int konfiguracja = 0;
    double minOdl = HUGE_VAL;
    SGLIB_LIST_GET_LAST(listaPunktow<T>, lista, next_point, pkoncowy);
    SGLIB_LIST_MAP_ON_ELEMENTS(listaPunktow<T>, lista->next_list, x, next_list, \
        if (odl2(lista->p, x->p)<minOdl) {minOdl=odl2(lista->p, x->p); konfiguracja = 1; dolaczanyCiag = x;})
    SGLIB_LIST_MAP_ON_ELEMENTS(listaPunktow<T>, lista->next_list, x, next_list, \
        if (odl2(pkoncowy->p, x->p)<minOdl) {minOdl=odl2(pkoncowy->p, x->p); konfiguracja = 2; dolaczanyCiag = x;})
    SGLIB_LIST_MAP_ON_ELEMENTS(listaPunktow<T>, lista->next_list, x, next_list, \
        SGLIB_LIST_GET_LAST(listaPunktow<T>, x, next_point, pk); \
        if (odl2(pkoncowy->p, pk->p)<minOdl) {minOdl=odl2(pkoncowy->p, pk->p); konfiguracja = 4; dolaczanyCiag = x;})
    SGLIB_LIST_MAP_ON_ELEMENTS(listaPunktow<T>, lista->next_list, x, next_list, \
        SGLIB_LIST_GET_LAST(listaPunktow<T>, x, next_point, pk); \
        if (odl2(lista->p, pk->p)<minOdl) {minOdl=odl2(lista->p, pk->p); konfiguracja = 3; dolaczanyCiag = x;})
    assert(konfiguracja <=4 && konfiguracja >= 1);

    SGLIB_LIST_DELETE(listaPunktow<T>, lista, dolaczanyCiag, next_list); // usunięcie ciągu dolaczanyCiag z listy
    dolaczanyCiag->next_list = NULL; // kosmetyka
    switch (konfiguracja) {
      case 1:
          SGLIB_LIST_REVERSE(listaPunktow<T>, dolaczanyCiag, next_point);
          dolaczanyCiag->next_list = lista->next_list; lista->next_list = NULL;
          SGLIB_LIST_ADD(listaPunktow<T>, lista, dolaczanyCiag, next_point);
          break;
      case 2:
        pkoncowy->next_point = dolaczanyCiag;
        break;
      case 3:
        dolaczanyCiag->next_list = lista->next_list; lista->next_list = NULL;
        SGLIB_LIST_ADD(listaPunktow<T>, lista, dolaczanyCiag, next_point);
        break;
      case 4:
        SGLIB_LIST_REVERSE(listaPunktow<T>, dolaczanyCiag, next_point);
        pkoncowy->next_point = dolaczanyCiag;
        break;
    }
    SGLIB_LIST_LEN(listaPunktow<T>, lista, next_list, liczbaSkladowych);
  }
  return lista;
}


// Uwaga funkcja zwraca listę, której elementy wskazują na pewne obszary
// tablicy p. Tablicy p nie można zatem zwolnić dopóki wykorzystuje się listę.
template <typename T>
listaPunktow<T> *order(
    T *p,//< początek tablicy ze współrzędnymi (punkty do ustawienia w ciągi)
              //  każdy punkt reprezentowany jest przez dim współrzędnych,
    int num,  //< liczba punktów (każdy po dim współrzędnych)
    T r, //< parametr funkcji choose (różne znaczenie w zależności od implementacji).
    int dim, // wymiar danych
    void *pozostale   //< wskaźnik do struktury zawierającej pozostałe argumenty funkcji (w razie potrzeby)
)
{
  listaPunktow<T> *lista = NULL, *tl;
  assert(num > 0);
  // włożenie punktów na listę
  while(num-- > 0) {
    tl = malloc(sizeof(listaPunktow<T>)); 
    assert(tl != NULL);
    tl->next_point = NULL;
    tl->p = p;
    SGLIB_LIST_ADD(listaPunktow<T>, lista, tl, next_list);
    p += dim;
  }

  // łączenie punktów odległych o nie więcej niż r
  tl = lista;
  double r2 = r*r;
  while( tl != NULL) 
  {
    bool kontynuacja = true;
    while (kontynuacja) 
    {
      kontynuacja = false;
      listaPunktow<T> *pkoncowy;
      SGLIB_LIST_GET_LAST(listaPunktow<T>, tl, next_point, pkoncowy);
      for(listaPunktow<T> *ta = tl->next_list; ta != NULL; ta = ta->next_list) 
      {
        if (odl2(ta->p, tl->p) <= r2) 
        {
          // usunięcie ciągu ta z listy ciągów
          SGLIB_LIST_DELETE(listaPunktow<T>, tl, ta, next_list); 
          ta->next_list = NULL; // kosmetyka
          // odwrócenie kolejności elementów ciągu ta
          SGLIB_LIST_REVERSE(listaPunktow<T>, ta, next_point);
          // dodanie ciągu ta za ciągiem tl
          SGLIB_LIST_ADD(listaPunktow<T>, tl->next_list, ta , next_list);
          // usunięcie tl z listy ciągów 
          SGLIB_LIST_DELETE(listaPunktow<T>, lista, tl, next_list);
          tl->next_list = NULL; //kosmetyka
          // dołączenie ciągu tl na koniec ta
          SGLIB_LIST_CONCAT(listaPunktow<T>, ta, tl, next_point);
          tl = ta;
          kontynuacja = true;
          break;
        } 
        else if (odl2(ta->p, pkoncowy->p)<= r2) 
        {
          // usunięcie ciągu ta z listy ciągów
          SGLIB_LIST_DELETE(listaPunktow<T>, tl, ta, next_list); 
          ta->next_list = NULL; // kosmetyka
          // dołączenie ciągu ta na koniec tl
          SGLIB_LIST_CONCAT(listaPunktow<T>, tl, ta, next_point);
          kontynuacja = true;
          break;
        }
      }
    }
    tl = tl->next_list;
  }
  return lista;
}

#endif // CPU_MRUP_ORDER_HPP
