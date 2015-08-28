function [centre,cluster] = kmean( testSet,k )
%KMEAN Summary of this function goes here
%   Detailed explanation goes here
[m,n]=size(testSet);
centre=zeros(k,n);
x=testSet;
for i=1:n
   mini=min(x(:,i)); 
   maxi=max(x(:,i)); 
   rangei=maxi-mini; 
   centre(:,i)=mini+rand(k,1)*rangei; 
end
loop=1;
cluster=zeros(m,2);
while loop
   loop=0;
   for j=1:m
      mindist=inf;minindex=-1; 
      for num1=1:k 
          dist=com_dist(centre(num1,:),x(j,:));
          if dist < mindist
             mindist=dist; 
             minindex=num1;
              
          end
      end
       if(cluster(j,1) ~= minindex)
           loop = 1;
           
       end
       
       cluster(j,:)=[minindex,mindist.^2];
    end
    for num2=1:k
         num3=find(cluster(:,1)==num2);
         x1=x(num3,:);
         centre(num2,:)=mean(x1);
        
    
    
    
    
    end
    
    
    
    
    
end     

end

