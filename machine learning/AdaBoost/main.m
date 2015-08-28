clear
clc
w=[];
D=ones(299,1);
for num1=1:299
   D(num1)=1.0/299; 
    
end
a=zeros(1,10);
for num2=1:10
    main2;
    d=D;
    w=[w;weights'];
    for num3=1:length(e)
        p(num3)=D(e(num3));
        P=sum(p);
    end
    a(num2)=0.5*log((1-P)/P);
    num=e;
    for num4=1:299
       if y(num4)== y(num4)
            D(num4)=D(num4)*exp(-a(length(a)));
       else
           D(num4)=D(num4)*exp(a(length(a)));
       end
    end
    for num5=1:299
        D(num5)=sum(D);
    end
    
    
end
