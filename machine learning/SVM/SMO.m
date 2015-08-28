classdef SMO <handle
    %SMO Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        testdata;
        traindata;
        testlabel;
        trainlabel;
        c;
        tol;
        m;
        alpha;
        b;
        ecache;
    end
    
    methods
        function obj=SMO(data,label,c,tol)
            obj.traindata=data;
            obj.trainlabel=label;
            obj.c=c;
            obj.tol=tol;
            obj.m=size(obj.traindata,1);
            obj.alpha=zeros(obj.m,1);
            obj.b=0;
            obj.ecache=zeros(obj.m,2);
        end
 %       function init(obj)
 %           obj.m=size(obj.traindata,1);
 %           obj.alphas=zeros(obj.m,1);
 %           obj.b=0;
 %           obj.ecache=zeros(obj.m,2);
 %       end
        function deta=calcek(obj,k)
            fx=(obj.alpha.*obj.trainlabel)'*(obj.traindata*obj.traindata(k,:)')+obj.b;
            deta=fx-obj.trainlabel(k);
        end  
        function [maxk,ej]=selectJ(obj,i,ei)
            maxk=0;
            maxdeta=0;
            ej=0;
            obj.ecache(i,:)=[1,ei];
            list=find(obj.ecache(:,1)>0);
            num=length(list);
            if(num > 1)
               for h=1:num 
                   k=list(h);
                   if k==i
                       continue;
                   end
                   ek=obj.calcek(k);
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
                  maxk=randi([1,obj.m],1,1);
                  ej=obj.calcek(maxk);
               end
            end
            
        end
        function x=select_alpha(obj,H,L,alpha)
            if(L>H)
                x=-1;
                return;
            end
            if(alpha>=H)
                x=H;
            elseif(alpha <= L)
                x=L;
            else
                x=alpha;
            end
        end
        function x=updataek(obj,k)
            ek=obj.calcek(k);
   %         obj.ecache(k,:)=[1,ek];
            x=[1,ek];
        end
        function r=inner(obj,i)
            ei=obj.calcek(i);
  %          r=0;
            if ((obj.trainlabel(i)*ei < -obj.tol) && (obj.alpha(i)<obj.c)) || ((obj.trainlabel(i)*ei > obj.tol) && (obj.alpha(i)>0))
                [j,ej]=obj.selectJ(i,ei);
                alphaIold=obj.alpha(i);
                alphaJold=obj.alpha(j);
                if(obj.trainlabel(i) == obj.trainlabel(j))
                    L=max(0,obj.alpha(i)+obj.alpha(j)-obj.c);
                    H=min(obj.c,obj.alpha(i)+obj.alpha(j));
                else
                    L=max(0,obj.alpha(j)-obj.alpha(i));
                    H=min(obj.c,obj.alpha(j)-obj.alpha(i)+obj.c); 
                end
                if L==H
                    r=0;
                    fprintf('L=H\n') 
                    return ;
                end
                eta=2*(obj.traindata(i,:)*obj.traindata(j,:)')-(obj.traindata(i,:)*obj.traindata(i,:)')-(obj.traindata(j,:)*obj.traindata(j,:)');
                if eta >= 0
                    r=0;
                    fprintf('eta >= 0');
                    return;
                end
                obj.alpha(j)=obj.alpha(j)-obj.trainlabel(j)*(ei-ej)/eta;
                obj.alpha(j)=obj.select_alpha(H,L,obj.alpha(j));
                 obj.ecache(j,:)=obj.updataek(j);
                if(abs(obj.alpha(j)-alphaJold) < 0.0001)
                   r=0;
                   fprintf('j not moving enough\n');
                   return;
                end
                s=obj.trainlabel(i)*obj.trainlabel(j);
                obj.alpha(i)=obj.alpha(i)+s*(alphaJold-obj.alpha(j));
                obj.ecache(i,:)=obj.updataek(i);
                b1=obj.b-ei-obj.trainlabel(i)*(obj.alpha(i)-alphaIold)*(obj.traindata(i,:)*obj.traindata(i,:)')-obj.trainlabel(j)*(obj.alpha(j)-alphaJold)*(obj.traindata(i,:)*obj.traindata(j,:)');
                b2=obj.b-ej-obj.trainlabel(i)*(obj.alpha(i)-alphaIold)*(obj.traindata(i,:)*obj.traindata(j,:)')-obj.trainlabel(j)*(obj.alpha(j)-alphaJold)*(obj.traindata(j,:)*obj.traindata(j,:)');
                if(obj.alpha(i)>0) && (obj.alpha(i) < obj.c)
                           obj.b=b1;
                elseif(obj.alpha(j)>0) && (obj.alpha(j) < obj.c)
                    obj.b=b2;
                else
                    obj.b=(b1+b2)/2;
                end
                r=1;
                return;
                
            else
                r=0;
                return;
            end
        end
        function train(obj,maxiter)
            iter=0;
            entire=1;
            changer=0;
 %           tem=[];
            while((iter < maxiter) &&((changer > 0) || (entire)))
                changer=0;
                if entire
                    for i=1:obj.m
                       
                        num=obj.inner(i);
                         fprintf('num = %d\n',num);
                        changer=changer+num;
                        
                    end
                    iter=iter+1;
                else
                     tem=find(obj.alpha>0 &obj.alpha < obj.c);
                     l=length(tem);
                     for j=1:l
                        k=tem(j);
                         
                        num1=obj.inner(k);
                        fprintf('num1 = %d\n',num1);
                        changer=changer+num1;
                        
                     end
                     iter=iter+1;
                end
                if entire
                   entire=0; 
                elseif changer==0
                    entire=1;
                end
            end
        end
    end
    
end

