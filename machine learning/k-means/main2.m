    [m,n]=size(testSet);
     cent0=mean(testSet);
     centlist=cent0;
     cluster=zeros(m,n);
     k=4;
     for i=1:m
        cluster(i,2)=com_dist(cent0,testSet(i,:)); 
        cluster(i,1)=1;
     end
    s=size(centlist);
     while (s(1) < k )
         minsse=inf;
         s=size(centlist);
         for j=1:s(1)
             num=find(cluster(:,1)==j);
             num1=find(cluster(:,1) ~= j);
             x=testSet(num,:);
             [cent,clus]=kmean(x,2);
             ssesplit=sum(clus(:,2));
             ssenosplit=sum(cluster(num1,2));
             if (ssesplit+ssenosplit) < minsse
                 bestsplit=j;
                 bestnewcens=cent;
                 bestcluster=clus;
             end
         end
         num2=find(bestcluster(:,1)==1);
         num3=find(bestcluster(:,1)==2);
         bestcluster(num2,1)=s(1)+1;
         bestcluster(num3,1)=bestsplit;
         centlist(bestsplit,:)=bestnewcens(1,:);
         centlist=[centlist;bestnewcens(2,:)];
         for h=1:length(num)
            cluster(num(h),:)=bestcluster(h,:); 
             
         end
     end