#include <stdio.h>
#include <stdlib.h>
#include <time.h>



int score;
int meilleur_score;
int nb_cases_libre;

int rand_a_b(int a, int b);
void pop_up(int** GRILLE);
int** init_grille();
void afficher_grille(int** GRILLE);


int main(){
  srand(time(NULL));
  int**  GRILLE=NULL;
  GRILLE=init_grille();
  afficher_grille(GRILLE);
  return 0;
}


/*Fonction générant un nombre random entre a et b*/

int rand_a_b(int a, int b){
  return rand()%(b-a) +a;
}


/*Fonction faisant apparaitre un 2 (7 chances sur 8) ou un 4 (1 chance sur 8) au hasard dans la GRILLE*/

void pop_up(int** GRILLE){
  int num;
  int k=rand_a_b(0,20);
  if(k<18){
    num=2;
  }
  else
    num=4;
  int i=0;
  int j=0;
  while(k!=0){
    if(GRILLE[i][j]==0){
      k--;
    }
    i++;
    if(i==4){
      i=0;
      j++;  
    }
    if(j==4)
      j=0;
  }
  GRILLE[i][j]=num;
}


/*Fonction initialisant la GRILLE avec deux nombres (2 ou 4) et des 0 ailleurs*/

int** init_grille(){
  int** GRILLE=(int**)malloc((4)*sizeof(int*));
  int i,j;
    for(i=0;i<4;i++){
	GRILLE[i]=(int*)malloc((4)*sizeof(int));
    }
  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
      GRILLE[i][j]=0;
    }
  }
  pop_up(GRILLE);
  nb_cases_libre--;
  pop_up(GRILLE);
  nb_cases_libre--;
  return GRILLE;
}


/*Fonction affichant la GRILLE*/

void afficher_grille(int** GRILLE){
  int i,j;
  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
      printf("| %d ",GRILLE[i][j]);
    }
    printf("|\n");
  }
}
