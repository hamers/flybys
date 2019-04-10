all: flybyslibrary.c
	$(CC) -fPIC -shared -O3 -o flybyslibrary.so -lm flybyslibrary.c 
clean: 
	$(RM) flybyslibrary.so
	
	
