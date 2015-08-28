load('x_y')
[m,n]=size(x);
y_p=ones(1,m);
k=0.01;
w=[];
for i=1:m
   [y_p(i),w]=lwlr(x(i,:),x,y,k);   
end