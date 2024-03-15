/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/ 
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/
/*************************************************************************/
/**   File:         example.c                                           **/
/**   Description:  Takes as input a file:                              **/
/**                 ascii  file: containing 1 data point per line       **/
/**                 binary file: first int is the number of objects     **/
/**                              2nd int is the no. of features of each **/
/**                              object                                 **/
/**                 This example performs a fuzzy c-means clustering    **/
/**                 on the data. Fuzzy clustering is performed using    **/
/**                 min to max clusters and the clustering that gets    **/
/**                 the best score according to a compactness and       **/
/**                 separation criterion are returned.                  **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department Northwestern University                   **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Sang-Ha  Lee											**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					Simplified for main functionality: regular k-means	**/
/**					clustering.											**/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/types.h>
#include <fcntl.h>
#include <omp.h>
#include "getopt.h"

#include "kmeans.h"
#include "checkpoint.h"
extern double wtime(void);
/*----< checkpoint() >---------------------------- */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <limits.h>
#include"checkpoint.h"
int restart_id;
void write_cp(int rank, int iter, void *data, int size){
    char filename[64];

    sprintf( filename, "cpk_directory/check_%d_%d", rank, iter);
    printf("write check_%d_%d \n", rank, iter);
    FILE *fp = fopen( filename, "ab" );
    assert( NULL != fp );
    fwrite( &size, sizeof(int), 1, fp );
    fwrite( data, size, 1, fp );
    fclose( fp );

    sprintf( filename, "cpk_directory/tmp_%d", rank );
    fp = fopen( filename, "wb" );
    assert( NULL != fp );
    fwrite( (char *)&iter, sizeof(int), 1, fp);
    printf("Write tmp_%d: the iter value is: %d \n", rank, iter);
    fclose( fp );
}

void read_cp(int rank, int *iter, int *location, void *data, int size){
    int local_iter;
    char filename[64];
    sprintf( filename, "cpk_directory/tmp_%d", rank);
    FILE *fp = fopen( filename, "rb" );
    assert( NULL != fp );
    fseek(fp, 0, SEEK_SET);
    fread( (char *)&local_iter, sizeof(int), 1, fp );
    printf("local_iter: %d\n",local_iter);
    printf("Read tmp_%d: the iter value is: %d \n", rank, local_iter);
    fclose( fp );
    sprintf( filename, "cpk_directory/check_%d_%d", rank, local_iter);
//// testing ////
    printf("read check_%d_%d \n", rank, local_iter);
    fp = fopen( filename, "rb" );
    assert( NULL != fp );
    fseek(fp, *location, SEEK_SET);
    fread( &size, sizeof(int), 1, fp );
    *location += sizeof(int);
    fread( data, size, 1, fp );
    *location += size;
    fclose( fp );
    printf("Rank %d READCP file\n", rank);
    *iter = local_iter + 1;
}
/* ============================================ */
/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0) {
    char *help =
        "Usage: %s [switches] -i filename\n"
        "       -i filename     :  file containing data to be clustered\n"
        "       -b                 :input file is in binary format\n"
		"       -k                 : number of clusters (default is 8) \n"
        "       -t threshold    : threshold value\n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv) {
           int     opt;
    extern char   *optarg;
    extern int     optind;
           int     nclusters=5;
           char   *filename = 0;           
           float  *buf;
           float **attributes;
           float **cluster_centres=NULL;
           int     i, j;           
		   
           int     numAttributes;
           int     numObjects;           
           char    line[1024];
           int     isBinaryFile = 0;
           int     nloops;
           float   threshold = 0.001;
		   double  timing;


	while ( (opt=getopt(argc,argv,"i:k:t:b:r"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'b': isBinaryFile = 1;
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'k': nclusters = atoi(optarg);
                      break;
            case '?': usage(argv[0]);
                      break;
            default: usage(argv[0]);
                      break;
        }
    }

    if (filename == 0) usage(argv[0]);

    numAttributes = numObjects = 0;

    /* from the input file, get the numAttributes and numObjects ------------*/
   
    if (isBinaryFile) {
        int infile;
        if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        read(infile, &numObjects,    sizeof(int));
        read(infile, &numAttributes, sizeof(int));
   

        /* allocate space for attributes[] and read attributes of all objects */
        buf           = (float*) malloc(numObjects*numAttributes*sizeof(float));
        attributes    = (float**)malloc(numObjects*             sizeof(float*));
        attributes[0] = (float*) malloc(numObjects*numAttributes*sizeof(float));
        for (i=1; i<numObjects; i++)
            attributes[i] = attributes[i-1] + numAttributes;

        read(infile, buf, numObjects*numAttributes*sizeof(float));

        close(infile);
    }
    else {
        FILE *infile;
        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        while (fgets(line, 1024, infile) != NULL)
            if (strtok(line, " \t\n") != 0)
                numObjects++;
        rewind(infile);
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                /* ignore the id (first attribute): numAttributes = 1; */
                while (strtok(NULL, " ,\t\n") != NULL) numAttributes++;
                break;
            }
        }
     

        /* allocate space for attributes[] and read attributes of all objects */
        buf           = (float*) malloc(numObjects*numAttributes*sizeof(float));
        attributes    = (float**)malloc(numObjects*             sizeof(float*));
        attributes[0] = (float*) malloc(numObjects*numAttributes*sizeof(float));
        for (i=1; i<numObjects; i++)
            attributes[i] = attributes[i-1] + numAttributes;
        rewind(infile);
        i = 0;
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue; 
            for (j=0; j<numAttributes; j++) {
                buf[i] = atof(strtok(NULL, " ,\t\n"));
                i++;
            }
        }
        fclose(infile);
    }
 	nloops = 1; 
	printf("I/O completed\n");

	memcpy(attributes[0], buf, numObjects*numAttributes*sizeof(float));
	memcpy(attributes[0], buf, numObjects*numAttributes*sizeof(float));
    int rw_ranks = 0;
    int rw_location = 0;
    i = 0;
    restart_id = atoi(argv[3]);
    printf("%d restartid\n",restart_id);
    if (restart_id == 1) {
                cluster_centres = (float**)malloc(nclusters * sizeof(float*));
        cluster_centres[0] = (float*)malloc(nclusters * numAttributes * sizeof(float));
	    read_cp(rw_ranks, &i, &rw_location, cluster_centres, numAttributes* nclusters*sizeof(float));
    	printf("i == %d\n",i);
    }
    printf("nloops == %d\n",nloops);
	timing = omp_get_wtime();
    for (i=0; i<nloops; i++) {
        		
        cluster_centres = NULL;
        cluster(numObjects,
                numAttributes,
                attributes,           /* [numObjects][numAttributes] */
                nclusters,
                threshold,
                &cluster_centres   
               );
            if (0 == restart_id && i  == 0) {
                write_cp(rw_ranks, i, cluster_centres, numAttributes* nclusters * sizeof(float));
                printf("i == %d,,\n",i);
		exit(0);
            }  
	//printf("%zu,%zu\n",sizeof(cluster_centres),sizeof(float**));
	//printf("%d ,%d\n",nclusters,numAttributes);
     
    }
    timing = omp_get_wtime() - timing;

	printf("number of Clusters %d\n",nclusters); 
	printf("number of Attributes %d\n\n",numAttributes); 
    printf("Cluster Centers Output\n"); 
	printf("The first number is cluster number and the following data is arribute value\n");
	printf("=============================================================================\n\n");
	
    for (i=0; i<nclusters; i++) {
		printf("%d: ", i);
        for (j=0; j<numAttributes; j++)
            printf("%f ", cluster_centres[i][j]);
        printf("\n\n");
    }
	printf("Time for process: %f\n", timing);

    free(attributes);
    free(cluster_centres[0]);
    free(cluster_centres);
    free(buf);
    return(0);
}

