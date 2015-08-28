load('horse_train')
%load('horse_test')
[m,n]=size(traindata);
weights=ones(n,1);
a=0.001;
while j<500
    h=sigmoid(traindata*weights);
    error=trainlabel-h';
    weights=weights+a*(trainlabel'*error);
    j=j+1;
    
    
    
    
    
    
end