// przetwarzanie danych pobranych ze standardowego wejścia (w tekstowym formacie)
#define _XOPEN_SOURCE 500
#include<gsl/gsl_rng.h>
#include<gsl/gsl_randist.h>
#include<gsl/gsl_statistics_double.h>
#include<gsl/gsl_sort.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h> // atof, malloc, ...
#include<unistd.h> // sleep, getopt
#include<string.h>
#include<assert.h>
#include"choose.h"
#include"evolve.h"
#include"decimate.h"
#include"order.h"
#include"sglib.h"

#define IND_DIAG(k,n) (((k)*(2*(n) + 1 - (k)))/2)

int dim = 2;
static gsl_rng * rng = NULL; 

void drukuj_punkty(FILE *f, double *p, int num) {
  int i=0;
  while(num-- > 0) {
    while(i++ < dim) 
      fprintf(f, "%9.4f ",*(p++));
    i = 0;
    fprintf(f,"\n");
  }
}

//* wydruk do pliku współrzędnych punktów należących do składowej spójności
//* jeśli a i b nie są jednocześnie zerami, to w dodatkowej kolumnie drukowana
//jest liczba a * x + b, gdzie x to bieżący numer punktu
void drukuj_skladowa(
    FILE *f,  //< uchwyt do pliku
    double *p,//< początek tablicy ze współrzędnymi
    int num,  //< liczba punktów (każdy po dim współrzędnych)
    int *mi,  //< "macierz incydencji", na "przekątnej" numery składowej
    int skl,  //< numer składowej do pokazania
    double a,
    double b
);

void drukuj_skladowa(FILE *f, double *p, int num, int *mi, int skl, double a, double b) {
  for(int k=0; k < num; k++) {
    if (skl != 0) {
      if (skl > 0 && mi[IND_DIAG(k,num)] != skl) continue;
      if (skl < 0 && mi[IND_DIAG(k,num)] == -skl) continue;
    }
    for(int d=0; d < dim; d++) 
      fprintf(f, "%9.4f ", p[k * dim + d]);
    if (a != 0 || b!= 0)
      fprintf(f, "%9.4f", k * a + b);
    fprintf(f, "\n");
  }
}

void drukuj_ciagi(FILE *f, listaPunktow *lista) {
//	listaPunktow *lc;
//	listaPunktow *lp;
	fprintf(f,"#== spójne ciągi ==#\n");
	SGLIB_LIST_MAP_ON_ELEMENTS(listaPunktow, lista, lc, next_list, \
		SGLIB_LIST_MAP_ON_ELEMENTS(listaPunktow, lc, lp, next_point, \
			drukuj_punkty(f, lp->p, 1); \
			); \
		fprintf(f,"\n");
	);
	fprintf(f,"#------------------#\n");
}

void pokaz_punkty(FILE *f, double *p, int num) {
  assert(f != NULL);
  fprintf(f, "plot "
      "'-' w p pt 7 lc rgb 'red' ps 0.2, "
      "\n");
  drukuj_punkty(f, p, num);
  fprintf(f,"e\n");
  fflush(f);
}
void pokaz_kolka(FILE *f, double *p, int num, double *k, int licz, double r) {
  assert(f != NULL);
//  fprintf(f, "set terminal pngcairo notransparent\n");
//  fprintf(f, "set out 'tmp.png'\n");
//  fprintf(f, "set key off\n"); // bez legendy
//  fprintf(f, "set size ratio -1\n"); // równe skale na osiach
//  fprintf(f, "set grid xtics ytics\n");
  fprintf(f, "plot "
      "'-' w p pt 7 lc rgb 'red' ps 0.2, "
      "'-' w circles lc rgb 'blue' fs transparent solid 0.15 noborder,"
      "'-' u 1:2 w p pt 7 lc rgb 'black'\n");
  drukuj_punkty(f, p, num);
  fprintf(f,"e\n");
  drukuj_skladowa(f, k, licz, NULL, 0, 0, r);
  fprintf(f,"e\n");
  drukuj_punkty(f, k, licz);
  fprintf(f,"e\n");
  fflush(f);
}

void pokaz_ciagi(FILE *f, double *p, int num, listaPunktow *lista) {
  assert(f != NULL);
//  fprintf(f, "set terminal pngcairo notransparent\n");
//  fprintf(f, "set out 'tmp.png'\n");
//  fprintf(f, "set key off\n"); // bez legendy
//  fprintf(f, "set size ratio -1\n"); // równe skale na osiach
//  fprintf(f, "set grid xtics ytics\n");
  fprintf(f, "plot "
      "'-' w p pt 7 lc rgb 'red' ps 0.2");
	int i;
	SGLIB_LIST_LEN(listaPunktow, lista, next_list, i);
	while (i-- > 0) 
		fprintf(f, ", '-' u 1:2 w linespoints pt 7 lw 2 lc rgb 'black'");
	fprintf(f,"\n");
  drukuj_punkty(f, p, num);
  fprintf(f,"e\n");
	SGLIB_LIST_MAP_ON_ELEMENTS(listaPunktow, lista, lc, next_list, \
		SGLIB_LIST_MAP_ON_ELEMENTS(listaPunktow, lc, lp, next_point, \
			drukuj_punkty(f, lp->p, 1); \
			); \
		fprintf(f,"e\n");
	);
  fflush(f);
}

void dodaj_szum(double *p, int num, int d, double sigma) {
  // dodatnie do num elementów tablicy (każdy po d doubli) 
  // spod adresu p wartości losowych z rozkładu normalnego 
  // z zerową średnią i odch. st. sigma
  num *= d;
  while(num-- > 0) *(p++) += gsl_ran_gaussian(rng, sigma);
}

// odległości punktów od odcinka [0,1] x {0}^(dim-1)
// Może jednak liczyć odległość od prostej [-inf,+inf] x {0}^(dim-1)???
// bo maks może wypadać "z brzegu".
void odl_do_wzorca(double *p, int num, double *mediana, double *srednia, double *maks)
{
	double *odl;
	assert( (odl = calloc(num, sizeof(double))) != NULL);

	for(int i=0; i < num; i++) {
		if (*p < 0) odl[i] += *p * *p;	// 1-sza współrzędna
		else {
			double t = *p - 1;
			if (t > 0) odl[i] += t * t;
		}
		p++;
		for(int k=1; k<dim; k++, p++)
			odl[i] += *p * *p;
	}

	gsl_sort(odl, 1, num);
	*mediana = sqrt(gsl_stats_median_from_sorted_data(odl, 1, num)); 
	*maks = sqrt(odl[num-1]);
	*srednia = sqrt(gsl_stats_mean(odl, 1, num));
	free(odl);
}

int por_punkty(const void * a, const void * b)
{
	if ( *(double*)a < *(double*)b ) return -1;
	if ( *(double*)a > *(double*)b ) return 1;
	a = (void *) ((double*)a + 1);
	b = (void *) ((double*)b + 1);
	if ( *(double*)a < *(double*)b ) return -1;
	if ( *(double*)a > *(double*)b ) return 1;
	return 0;
}

//Uwaga poniższa funkcja zmienia współrzędne punktów z tablicy p
void odl_od_wzorca(double *p, int num, double *brzeg, double *wnetrze)
// funkcja wyznacza minimalną odległość między punktami z tablicy p, a punktami 
// {(0,0), (1,0)} (do brzeg) oraz maksymalną odległośc między punktem z odcinka
// [0,1] x {0} a najbliższym punktem z tablicy p.
{
//	printf("-------\n");
//	drukuj_punkty(stdout, p, num);
	for(int i=0; i < num; i++) {
		int k = 2;
		for(p[i*dim + 1] *= p[i*dim + 1]; k<dim; k++)
			  p[i*dim + 1] += (p[i*dim + k] * p[i*dim + k]);
		p[i*dim + 1] = sqrt(p[i*dim + 1]);
	}
	qsort((void *) p, num, dim * sizeof(double), por_punkty);
//	printf("-------\n");
//	drukuj_punkty(stdout, p, num);

	double xp, yp;
	// wyszukanie punktu najbliższego (0,0)
	{
		int ind = 0;
		double *pt = p;
		xp = pt[0];
		yp = pt[1];
		pt += dim;
		*brzeg = xp * xp + yp * yp; // kw. odl. od (0,0)
		for(int i = 1; i < num; i++) {
			double o;
			o = pt[0] * pt[0] + pt[1] * pt[1]; // kw. odl. od (0,0)
			if (o < *brzeg) {
				ind = i;
				*brzeg = o;
			}
			pt += dim;
		}
		p += ind * dim;
		xp = p[0];
		yp = p[1];
		p += dim;
		num -= (ind+1);
	}

	*wnetrze = 0;
	double xmax = 0; 
// printf("bliski 0: *brzeg = %8.5f, *wnetrze = %8.5f, xmax = %8.5f, xp=%8.5f, yp=%8.5f\n", 
// 		*brzeg, *wnetrze, xmax, xp, yp);

	while (num >= 0) {
		double xmax_new = 1.0;
		double xp_new=xp, yp_new=yp;
		// Szukamy przecięcia brzegu komórki związanaj z punktem (xp,yp) diagramu
		// Voronoi związanego z punktam (xp,yp), (p[0],p[1]), (p[2],p[3]), ...
		// z prostą y=0
		for(int i=0; i < num; i++) {
			double xtmp = (p[i*dim] * p[i*dim] + p[i*dim + 1] * p[i*dim + 1] - (xp*xp + yp*yp)) /
						(2 * (p[i*dim] - xp));
			if ( xtmp < xmax_new ) {
					xmax_new = xtmp;
					xp_new = p[i*dim    ];
					yp_new = p[i*dim + 1];
					p += (i+1) * dim;
					num -= (i+1);
					i = 0;
			}
		}
		xmax = xmax_new;
		double to = (xp - xmax)*(xp - xmax) + yp*yp;
		if (xmax_new == 1.0) {
//			printf("xmax_new=1.0\n");
			if (to > *brzeg) *brzeg = to;
			if (*wnetrze == 0) *wnetrze = *brzeg;
			break;
		} else {
			xp = xp_new;
			yp = yp_new;
			if (  to > *wnetrze ) *wnetrze = to;
		}
// printf("kolejny: *brzeg = %8.5f, *wnetrze = %8.5f, xmax = %8.5f, xp=%8.5f, yp=%8.5f\n", 
// 		*brzeg, *wnetrze, xmax, xp, yp);
	}
// printf("-------\n");
// printf("koniec: *brzeg = %8.5f, *wnetrze = %8.5f, xmax = %8.5f, xp=%8.5f, yp=%8.5f\n", 
// 		*brzeg, *wnetrze, xmax, xp, yp);
	*brzeg = sqrt(*brzeg);
	*wnetrze = sqrt(*wnetrze);
//	printf("*brzeg = %8.5f, *wnetrze = %8.5f\n", *brzeg, *wnetrze);
}



int main(int argc, char *argv[]){
  double *punkty;
  double *wybrane;

	double r1=0.1, r2=0.2, sigma=0;
  int n = 1000, powt = 1;
	char rdzen[256]; // rdzeń nazwy plików do zapisu

	FILE *f_gplot;
	FILE *f_stat;
	FILE *f_bin;
	char stat_naz[256];
	char bin_naz[256];
	int rysuj = 0, zapisuj = 0, randomizuj = 1;

	/*
	double pun[12] = {
		-.25, 0.25,
		0, 1,
		0.25, 0.5,
		0.5, -0.25,
		0.25, 0.3,
		0.95, -.1};

	double brzeg, wnetrze;
	odl_od_wzorca(pun, 6, &brzeg, &wnetrze);
	return 0;
	*/

//----------------------- wczytanie argumentów programu -----------------------
	{
		int opt;
		extern char *optarg;
		extern int optind, opterr, optopt;
		while ((opt = getopt(argc, argv, "p:n:r:R:s:gbl")) != -1) {
			switch (opt) {
				case 'n':
					n = atoi(optarg);
					break;
				case 'p':
					powt = atoi(optarg);
					break;
				case 'r':
					r1 = atof(optarg);
					break;
				case 's':
					sigma = atof(optarg);
					break;
				case 'l':
					randomizuj = 0;
					break;
				case 'R':
					r2 = atof(optarg);
					break;
				case 'g':
					rysuj = 1;
					break;
				case 'b':
					zapisuj = 1;
					break;
				default: /* '?' */
					fprintf(stderr, 
"Wywołanie: %s [-n l_probek] [-p l_powtorzen] [-r r] [-R R] [-s odch_std] [-gb] rdzen\n"
"\t b -- zapisywanie plików binarnych z punktami wejściowymi i wyjściowymi\n"
"\t g -- rysowanie (do plików png)\n",
							argv[0]);
					exit(EXIT_FAILURE);
			}
		}
		printf("n=%3d; powt=%3d; r1=%8.3f; r2=%8.3f; sigma=%8.3f\n", n, powt, r1, r2, sigma);
		if (optind >= argc) {
			fprintf(stderr, "Potrzebny argument opcji\n");
			exit(EXIT_FAILURE);
		}
		strncpy(rdzen, argv[optind], 255);
	}

  assert( (rng=gsl_rng_alloc(gsl_rng_random_glibc2)) != NULL);

//--------------- randomizacja zarodka generatora liczb losowych ---------------
  if (randomizuj)
  { 
    FILE *randomData = fopen("/dev/urandom", "r");
    unsigned long int myRandomLongInt;
    assert(randomData != NULL);
    fread(&myRandomLongInt, sizeof myRandomLongInt, 1, randomData);
    gsl_rng_set(rng, myRandomLongInt);
    // you now have a random integer!
    fclose(randomData);
  }

	if (rysuj) {
		f_gplot = popen("gnuplot","w");
//	 	f_gplot = fopen("tmp.gnuplot","w");
		assert(f_gplot != NULL);
		fprintf(f_gplot, "set terminal pngcairo notransparent\n");
		fprintf(f_gplot, "set key off\n"); // bez legendy
		fprintf(f_gplot, "set size ratio -1\n"); // równe skale na osiach
		fprintf(f_gplot, "set grid xtics ytics\n");
//		fprintf(f_gplot, "set xrange [-0.2:1.2]\n");
//		fprintf(f_gplot, "set yrange [%f:%f]\n", -3*r1, 3*r1);
	}

	snprintf(stat_naz, 255, "%s_stat.txt", rdzen);
	f_stat = fopen(stat_naz, "a");
	assert(f_stat != 0);

//============================== wczytanie danych ==============================
	fprintf(stderr, "wczytywanie danych...\n");
	char line[1024];
	char buf[1024];
	assert(fgets(line, 1024, stdin) != NULL);
//	printf("line=<%s>\n", line);	
	memcpy((void *)buf, (void *)line, sizeof(line));
//-------- wyznaczenie wymiaru przestrzeni, z której pochodzą punkty ---------
	assert(strtok(buf, " \t") != NULL);
	dim = 1;
	while (strtok(NULL, " \t") != NULL) dim++;
	fprintf(stderr, "dim = %d\n", dim);
//----------------------------- wczytanie punktów -----------------------------
  punkty  = malloc(n * dim * sizeof(double));
  assert(punkty != NULL);
	char *tpc;
	int licznik_linii = 0;
	int i = 0; // indeks w tablicy punkty
	do {
//		printf("nl=%d, line=<%s>\n", licznik_linii, line);	
		licznik_linii++;
		if (licznik_linii > n) {
			n += 100;
			punkty = realloc(punkty, n * dim * sizeof(double));
			assert(punkty != NULL);
		}
		for(int k=0; k<dim; k++) {
			tpc = strtok((k==0)?line:NULL, " \t"); assert(tpc != NULL);
			punkty[i++] = atof(tpc);
		} 
//		drukuj_punkty(stderr, punkty+i-dim, 1);
		
	} while(fgets(line, 1024, stdin) != NULL);
	n = licznik_linii;
	punkty = realloc(punkty, n * dim *sizeof(double));
	assert(punkty != NULL);
	if (sigma > 0) {
		printf("przed dodaniem szumu...\n");
		dodaj_szum(punkty, n, dim, sigma);
	}


  wybrane = malloc(n * dim * sizeof(double));
  assert(wybrane != NULL);

//------------------------------------------------------------------------------
	int n_powt = 0;
	while (++n_powt <= powt) {
//----------------- przetasowanie punktów (zmiana kolejności) ----------------
		if (powt > 1) gsl_ran_shuffle(rng, punkty, n, dim * sizeof(double));
//	printf("po tasowaniu\n");

		if (zapisuj) {
			snprintf(bin_naz, 255, "%s_%03d.bin", rdzen, n_powt);
			f_bin = fopen(bin_naz, "w");
			assert(f_bin != NULL);
			fwrite(punkty, sizeof(double) * dim, n, f_bin);
		}

		int liczba = choose(punkty, n, wybrane, r1, NULL);
		printf("Wybrano reprezentantów, liczba=%d\n", liczba);
		//  getc(stdin);
		drukuj_punkty(stdout, wybrane, liczba);
    printf("---\n");

		int stara_liczba = 0;
		int iteracja = 0;
		while(stara_liczba != liczba) {
			stara_liczba = liczba;
			printf("Ewolucja nr %3d...\n", iteracja);
			evolve(punkty, n, r1, wybrane, liczba, NULL);
			if (rysuj) {
				fprintf(f_gplot, "set out '%s_%03d_%03d_a.png'\n", rdzen, n_powt, iteracja);
				pokaz_kolka(f_gplot, punkty, n, wybrane, liczba, r1);
			}
			printf("Decymacja nr %3d...\n", iteracja);
			wybrane = decimate(wybrane, &liczba, r2, NULL);
			if (rysuj) {
				fprintf(f_gplot, "set out '%s_%03d_%03d_b.png'\n", rdzen, n_powt, iteracja);
				pokaz_kolka(f_gplot, punkty, n, wybrane, liczba, r2);
//			getc(stdin);
			}
			iteracja ++;
		}

		listaPunktow *lista = order(wybrane, liczba, r2, NULL); 
		drukuj_ciagi(stdout, lista);

		if (rysuj) {
			fprintf(f_gplot, "set out '%s_%03d_in.png'\n", rdzen, n_powt);
			pokaz_punkty(f_gplot, punkty, n);
//			pokaz_kolka(f_gplot, punkty, n, wybrane, liczba, r1);
//			getc(stdin);
			fprintf(f_gplot, "set out '%s_%03d.png'\n", rdzen, n_powt);
			pokaz_ciagi(f_gplot, punkty, n, lista);
		}
		if (zapisuj) {
			fwrite(wybrane, sizeof(double) * dim, liczba, f_bin);
			fclose(f_bin);
		}
//		drukuj_punkty(stdout, wybrane, liczba);
		fprintf(f_stat, "%4d %4d %4d %5d %8.5f %8.5f %8.5f", dim, n_powt, powt, n, r1, r2, sigma);
		fprintf(f_stat, " %4d %4d", iteracja, liczba);
		{ 	
			int liczba_skladowych;
			SGLIB_LIST_LEN(listaPunktow, lista, next_list, liczba_skladowych);
			fprintf(f_stat, " %4d", liczba_skladowych);
		}
		{
			double mediana, sr, maks;
			odl_do_wzorca(wybrane, liczba, &mediana, &sr, &maks);
			fprintf(f_stat, " %8.5f %8.5f %8.5f", mediana, sr, maks);
		}
		{
			double brzeg, wnetrze;
			odl_od_wzorca(wybrane, liczba, &brzeg, &wnetrze);
			fprintf(f_stat, " %8.5f %8.5f", brzeg, wnetrze);
		}
		fprintf(f_stat,"\n");
		fflush(f_stat);
	}
	fclose(f_stat);
	free(wybrane);
	free(punkty);
	gsl_rng_free(rng);
	if (rysuj) {
		pclose(f_gplot);
	}
	return 0;
}
// vim: set ts=2, sw=2, expandtab
