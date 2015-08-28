load('horse_train')
load('horse_test')
[m,n]=size(traindata);
weights=ones(n,1);
R=zeros(1,299);


for j=1:500
    for k=1:299
       R(k)=k;   
    end
     r=RandomPermutation(R);
    for i=1:m
        a=4/(1+j+i)+0.01;
 %       r=RandomPermutation(R);
       num=r(i);
        a=0.001;
 %        num=unidrnd(m);
        h=sigmoid((traindata(num,:)*weights));
        error=trainlabel(num)-h;
        weights=weights+a*error*traindata(num,:)';
 %       traindata(r,:)=[];
 %       R=R(2:end);
    end
    
    
end
for k=1:m
 %   for q=1:length(weights)
 %       Y(k)=weights(q)*traindata(k,q);
 %       y(k)=1/(1+exp(-Y(k)));
 %       if y(k)>0.5
 %           y(k)=1;
 %       else 
 %           y(k)=0;
 %       end
  %  end
    Y(k)=traindata(k,:)*weights;
    y(k)=1/(1+exp(-Y(k)));
        if y(k)>0.5
            y(k)=1;
        else 
            y(k)=0;
        end    
end
accut=0;
error=0;
e=[];
for h=1:m
    if y(h)==trainlabel(h)
        accut=accut+1;
    else 
        error=error+1;
        e=[e,h];
    end
    
end
%p=error/(error+accut);

