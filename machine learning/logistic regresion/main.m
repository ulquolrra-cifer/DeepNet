load('data.mat')
[m,n]=size(data);
alpha=0.001;
epsiode=500;
weights = ones(n,1);
i=1;
%history=[];
while i<500
   h=sigmoid(data*weights);
 %  history=[history;h];
   error=label-h';
   weights=weights+alpha.*(data'*error);
    i=i+1;
end
