void fusion (int** GRILLE,int i1, int j1, int i2, int j2){
	/* cette fonction met la case deux dasn la case 1. si dasn la case 1 il y a une 0, on dacéle juste le nombre, sinon on le multiplie par deux*/
	
	/*si la fusion n'est pas possible*/
	if (GRILLE[i1][j1]!=0 && GRILLE[i1][j1]!=GRILLE[i2][j2]){
		printf("fusion imposible");
		return;
	}
	
	/*si on fait un dépalcement dasn une case vide*/
	if (GRILLE[i1][j1]!=0){
		GRILLE[i1][j1]=GRILLE[i2][j2];
		GRILLE[i2][j2]=0;
	}
	
	/* sinon fait une fusion avec une autre case et augmentation du score*/
	else {
		GRILLE[i1][j1]=GRILLE[i1][j1]+GRILLE[i2][j2];
		GRILLE[i2][j2]=0;
		SCORE +=  GRILLE[i1][j1];
	}
}

void deplacement_case_bas(int** GRILLE, int i, int j){
	/* cette fonction effectue si nessesaire le déplacement ou la fusion de la case placé en parametre*/
	int i1=i+1;
	if ((GRILLE[i1][j]!=0 && GRILLE[i][j]!=GRILLE[i1][j]) || i1>3){
		return;
	}else if (GRILLE[i][j]==GRILLE[i1][j]){
		fusion(GRILLE,i,j,i1,j);
		return;
	}else {
		fusion(GRILLE,i,j,i1,j);
		deplacement_case_bas(GRILLE,i1,j);
	}
}
	
	
void deplacement_case_haut(int** GRILLE, int i, int j){
	/* cette fonction effectue si nessesaire le déplacement ou la fusion de la case placé en parametre*/
	int i1=i-1;
	if ((GRILLE[i1][j]!=0 && GRILLE[i][j]!=GRILLE[i1][j]) || i1<0){
		return;
	}else if (GRILLE[i][j]==GRILLE[i1][j]){
		fusion(GRILLE,i,j,i1,j);
		return;
	}else {
		fusion(GRILLE,i,j,i1,j);
		deplacement_case_haut(GRILLE,i1,j);
	}
}

void deplacement_case_gauche(int** GRILLE, int i, int j){
	/* cette fonction effectue si nessesaire le déplacement ou la fusion de la case placé en parametre*/
	int j1=j-1;
	if ((GRILLE[i][j1]!=0 && GRILLE[i][j]!=GRILLE[i][j1]) || j1<0){
		return;
	}else if (GRILLE[i][j]==GRILLE[i][j1]){
		fusion(GRILLE,i,j,i,j1);
		return;
	}else {
		fusion(GRILLE,i,j,i,j1);
		deplacement_case_haut(GRILLE,i,j1);
	}
}

void deplacement_case_droite(int** GRILLE, int i, int j){
	/* cette fonction effectue si nessesaire le déplacement ou la fusion de la case placé en parametre*/
	int j1=j+1;
	if ((GRILLE[i][j1]!=0 && GRILLE[i][j]!=GRILLE[i][j1]) || j1>3){
		return;
	}else if (GRILLE[i][j]==GRILLE[i][j1]){
		fusion(GRILLE,i,j,i,j1);
		return;
	}else {
		fusion(GRILLE,i,j,i,j1);
		deplacement_case_haut(GRILLE,i,j1);
	}
}

void deplacement_bas(int **GRILLE){
	/* cette fonction effectue le parcours d'une grille placé en parametre*/
	int i,j;

	for(j=0; j<4; j++){
		for(i=3; i>=0; i--){
			deplacement_case_bas(GRILLE, i, j);
		}
	}
}

void deplacement_haut(int **GRILLE){
	/* cette fonction effectue le parcours d'une grille placé en parametre*/
	int i,j;

	for(j=0; j<4; j++){
		for(i=0; i<4; i++){
			deplacement_case_haut(GRILLE, i, j);
		}
	}
}

void deplacement_gauche(int **GRILLE){
	/* cette fonction effectue le parcours d'une grille placé en parametre*/
	int i,j;

	for(i=0; i<4; i++){
		for(j=3; j>=0; j--){
			deplacement_case_bas(GRILLE, i, j);
		}
	}
}

void deplacement_droit(int **GRILLE){
	/* cette fonction effectue le parcours d'une grille placé en parametre*/
	int i,j;

	for(i=0; i<4; i++){
		for(j=0; j<4; j++){
			deplacement_case_bas(GRILLE, i, j);
		}
	}
}

int rand_a_b(int a, int b){
	/*Fonction générant un nombre random entre a et b*/
	return rand()%(b-a) +a;
}


void pop_up(int** GRILLE){
  /*Fonction faisant apparaitre un 2 (7 chances sur 8) ou un 4 (1 chance sur 8) au hasard dans la GRILLE*/

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



int** init_grille(){
  /*Fonction initialisant la GRILLE avec deux nombres (2 ou 4) et des 0 ailleurs*/

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

void afficher_grille(int** GRILLE){
  /*Fonction affichant la GRILLE*/
	int i,j;
  	for(i=0;i<4;i++){
		for(j=0;j<4;j++){
	  	printf("| %d ",GRILLE[i][j]);
	}
	printf("|\n");
  	}
}

int etat_du_jeux(int** GRILLE){
	/*fonction qui envoie:
	- 0 jeux est en cours
	- 1 gagné
	- -1 perdu*/
	int i,j;

	/*Cas gagné*/
  	for(i=0;i<4;i++){
		for(j=0;j<4;j++){
	  		if (GRILLE[i][j] == 2048) return 1;
		}
	}

	/*Cas perdu*/
	if ((deplacement_possible_gauche != 1) && (deplacement_possible_droite != 1) 
	&& (deplacement_possible_bas != 1) && (deplacement_possible_haut != 1)){
		return -1;
	}

	/*Cas en cours*/
	return 0;

}