function y = sigmoid( x)
%SIGMOID Summary of this function goes here
%   Detailed explanation goes here
%    y=[];
     [m,n]=size(x);
        for i=1:m
           y(i)=1/(1+exp(-x(i)));
        end
    
       
    
end

