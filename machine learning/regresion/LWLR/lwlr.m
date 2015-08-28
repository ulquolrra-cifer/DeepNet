function [label,ws] = lwlr( point,x,y,k)
%LWLR Summary of this function goes here
%   Detailed explanation goes here
    
    [m,n]=size(x);
    weights=zeros(m,m);
    for j=1:m
       diff=point-x(j,:) ;
       weights(j,j)=exp((diff*diff')/(-2*(k^2))); 
        
        
    end
    a=inv(x'*weights*x);
    b=(x'*weights*y);
    ws=a*b;
    label=point*ws;
end

