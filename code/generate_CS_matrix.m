clc; clear all
N=256;
M=128;

A = ((rand(M,N)>0.5)*2-1)/sqrt(M);

save CSmatrix256128.mat A