//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <string.h>
#include <math.h>
//#include <malloc.h>
#include <stdlib.h>

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv) {
  FILE *f;
  char st1[max_size];
  char *bestw[N], *bestw2[N];
  char file_name[max_size], st[100][max_size];
  float dist, len, bestd[N], bestd2[N], vec[max_size], D;
  long long words, size, a, b, c, d, cn, bi[100];
  //char ch;
  float *M, *psum;
  char *vocab;
  int sample, nSamples;
  int *shuffle, *keyFeature;
  double epsilon, delta;
  int feature;
  srand(100);
  if (argc < 2) {
    printf("Usage: ./distance <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
  for (a = 0; a < N; a++) bestw2[a] = (char *)malloc(max_size * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  psum = (float *)malloc((long long)words * (long long)size * sizeof(float));
  shuffle = (int *)malloc(size * sizeof(int));
  keyFeature = (int *)malloc(words * sizeof(int));
  for (a = 0; a < size; a++) shuffle[a] = a;
  for (a = 0; a < size; a++) {
    b = rand() % (a+1);
    if (a != b) { shuffle[a] ^= shuffle[b]; shuffle[b] ^= shuffle[a]; shuffle[a] ^= shuffle[b]; }
  }
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);
  while (1) {
    for (a = 0; a < N; a++) { bestd[a] = 0; bestd2[a] = 0; }
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    printf("Enter word or sentence (EXIT to break), e.g., %s: ", vocab);
    a = 0;
    while (1) {
      st1[a] = fgetc(stdin);
      if ((st1[a] == '\n') || (a >= max_size - 1)) {
        st1[a] = 0;
        break;
      }
      a++;
    }
    if (!strcmp(st1, "EXIT")) break;
    cn = 0;
    b = 0;
    c = 0;
    while (1) {
      st[cn][b] = st1[c];
      b++;
      c++;
      st[cn][b] = 0;
      if (st1[c] == 0) break;
      if (st1[c] == ' ') {
        cn++;
        b = 0;
        c++;
      }
    }
    cn++;
    for (a = 0; a < cn; a++) {
      for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;
      if (b == words) b = -1;
      bi[a] = b;
      printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
      if (b == -1) {
        printf("Out of dictionary word!\n");
        break;
      }
    }
    if (b == -1) continue;
    printf("\n                                              Word       Cosine distance\n------------------------------------------------------------------------\n");
    for (a = 0; a < size; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {
      if (bi[b] == -1) continue;
      feature = 0;
      for (a = 0; a < size; a++) {
        vec[a] += M[a + bi[b] * size];
        if (fabs(M[a + bi[b] * size]) > fabs(M[feature + bi[b] * size])) feature = a;
      }
      keyFeature[bi[b]] = feature;
    }
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
    for (a = 0; a < N; a++) bestd[a] = -1;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    for (c = 0; c < words; c++) {
      a = 0; feature = 0;
      for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
      if (a == 1) continue;
      dist = 0;
      for (a = 0; a < size; a++) {
          dist += vec[a] * M[a + c * size];
          if (fabs(M[a + c * size]) > fabs(M[feature + c * size])) feature = a;
      }
      keyFeature[c] = feature;
      for (a = 0; a < N; a++) {
        if (dist > bestd[a]) {
          for (d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw[d], bestw[d - 1]);
          }
          bestd[a] = dist;
          strcpy(bestw[a], &vocab[c * max_w]);
          break;
        }
      }
    }
    for (a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);

    for (a = 0; a < N; a++) bestd2[a] = -1;
    for (a = 0; a < N; a++) bestw2[a][0] = 0;
    nSamples = 0; epsilon = 0.05; delta = 0.001;
    for (a = 0; a < words; a++) psum[a] = 0.0;
    printf("\n                                              Word       Approximation\n------------------------------------------------------------------------\n");
    while (nSamples < size){
      nSamples++;
      sample = shuffle[nSamples-1];
      for (c = 0; c < words; c++) {
        a = 0;
        for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
        if (a == 1) continue;
        psum[c] += vec[sample] * M[sample + c * size];
      }
      D = 0.01 * pow(size, 0.5) * pow((log(size) - log(delta))/(2.0*nSamples), 0.5);
      if (D < epsilon) break;
    }
    for (c = 0; c < words; c++) {
      psum[c] = psum[c]/nSamples*size;
      psum[c] += vec[keyFeature[bi[b]]] * M[keyFeature[bi[b]] + c * size];
      psum[c] += vec[keyFeature[c]] * M[keyFeature[c] + c * size];
      for (a = 0; a < N; a++) {
        if (psum[c] > bestd2[a]) {
          for (d = N-1; d > a; d--) {
            bestd2[d] = bestd2[d-1];
            strcpy(bestw2[d], bestw2[d-1]);
          }
          bestd2[a] = psum[c];
          strcpy(bestw2[a], &vocab[c * max_w]);
          break;
        }
      }
    }
    for (a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw2[a], bestd2[a]);
    printf("%lld\t%d\n", size, nSamples);
  }
  return 0;
}
