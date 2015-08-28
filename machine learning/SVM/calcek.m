function ek = calcek( data,label,alpha,b )
%CALCEK Summary of this function goes here
%   Detailed explanation goes here
   fx=(alpha.*label)'*(data*data(k,:)')+b;
   ek=fx-trainlabel(k);

end

