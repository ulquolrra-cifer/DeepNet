main2
w=weights;
y=zeros(100,1);
for i=1:100
   y(i)=w(1)*data(i,1)+data(i,2)*w(2)+data(i,3)*w(3); 
   y(i)=1/(1+exp(-y(i))); 
   if y(i)>0.5
       y(i)=1;
   else
       y(i)=0;
   end
end
a=0;
error=0;
e=[];
for i=1:100
    if(y(i)==label(i))
        a=a+1;
    else
        error=error+1;
        e=[e;i];
    end
    
    
end
