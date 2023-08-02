function [h] = generate_channel(N1,N2,L,type)

N=N1*N2;
d=0.5;
h=zeros(N,1);
alpha = (normrnd(0, 1, L, 1) + 1i*normrnd(0, 1, L, 1)) / sqrt(2);

if type==1 
    phi=pi*rand(1,L)-pi/2;
    for l = 1:L
        a = 1/sqrt(N)*exp(1i*2*pi*[-(N-1):2:N-1]'/2*d*sin(phi(l)));  
        h = h + alpha(l)*a;
    end
end

if type==2    
    phi1=pi*rand(1,L)-pi/2;
    phi2=pi*rand(1,L)-pi/2;
    for l = 1:L
        a1 = 1/sqrt(N1)*exp(-1i*2*pi*[-(N1-1):2:N1-1]'/2*d*sin(phi1(l))*sin(phi2(l)));
        a2 = 1/sqrt(N2)*exp(-1i*2*pi*[-(N2-1):2:N2-1]'/2*d*cos(phi2(l)));
        a=kron(a1,a2);
        h = h + alpha(l)*a;
    end
end

h=sqrt(N/L)*h;
end

