clc; clear all; close all
N1=16;
N2=16;
N=N1*N2;
L=3;
d=0.5;
type=2; 
S=80000;

%% ULA
if type==1
    UN=(1/sqrt(N))*exp(1i*2*pi*[-(N-1):2:N-1]'/2*d*[-(N-1):2:N-1]*(1/N));
end
%% UPA
if type==2
    UN1=(1/sqrt(N1))*exp(1i*2*pi*[-(N1-1):2:N1-1]'/2*d*[-(N1-1):2:N1-1]*(1/N1));
    UN2=(1/sqrt(N2))*exp(1i*2*pi*[-(N2-1):2:N2-1]'/2*d*[-(N2-1):2:N2-1]*(1/N2));
    UN=kron(UN1,UN2);
end
x=zeros(N,S); % beamspace channel
for s=1:S
    x(:,s)=UN.'*generate_channel(N1,N2,L,type);
end
save SV_UPA_training_channel.mat x
