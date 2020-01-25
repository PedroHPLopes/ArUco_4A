#include <stdio.h>
#include <stdlib.h>

int main(){
    while(1){
        if (getc() == '\033') { // if the first value is esc
            getc(); // skip the [
            switch(getc()) { // the real value
                case 'A':
                    printf("up");
                    // code for arrow up
                    break;
                case 'B':
                    printf("down");
                    break;
                case 'C':
                    printf("right");
                    break;
                case 'D':
                    printf("left");
                    break;
            }
        }   
    }
}
