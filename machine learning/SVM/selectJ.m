function [maxk,ej] = selectJ(ecache,i,ei,m,data,label,alpha,b )
%SELECTJ Summary of this function goes here
%   Detailed explanation goes here
      maxk=0;
            maxdeta=0;
            ej=0;
            ecache(i,:)=[1,ei];
            list=find(obj.ecache(:,1)>0);
            num=length(list);
            if(num > 1)
               for h=1:num 
                   k=list(h);
                   if k==i
                       continue;
                   end
                   ek=calcek(k);
                   delta=abs(ei-ek);
                   if delta>maxdeta
                       maxk=k;
                       maxdeta=delta;
                       ej=ek;
                   end
                   
               end
                
   %         elseif (num==1)
                
                
            else
   %            m=obj.m
   %            j=randint(1,1,[i,m]);
               maxk=i;
               while maxk==i
                  maxk=randi([1,m],1,1);
                  ej=calcek(maxk);
               end
            end

end

