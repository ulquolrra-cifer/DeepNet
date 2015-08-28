clear
clc
load('x_y')
a=inv(x'*x);
w=a*x'*y;
plot(x(:,2),y,'.');
z=w'*x';
hold on
plot(x(:,2),z)