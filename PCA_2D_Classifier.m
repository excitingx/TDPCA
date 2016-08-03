%--------------------------------------------------------------------------  
% ����˵����2DPCA�㷨  
% ���ղ���  
%   trainingSet:ѵ�����ݼ���height*width*num�ľ���  
%   testingSet:�������ݼ���3ά����  
%   numClass:ѵ�������Ͳ���������ӵ�е������  
%   thresthold:��ֵ��������������������ѡ�񣬼Ⱦ���dȡֵ��  
%              �����ݵ�threstholdΪС������0.95���Զ�������ֵ���㣬d,q��ȡֵ��  
%              �����ݵ�threstholdΪ>=1��������30,�����d=30,numShape=d  
%     
% ��������  
%     accuracy:����׼ȷ��  
%     right:��ȷ�����������Ŀ  
%   
%author:������  
%Date:2008-10-22  
%--------------------------------------------------------------------------  
  
function [accuracy,right,numShape]=PCA_2D_Classifier(trainingSet,testingSet,numClass,thresthold)  
     
    numTrainInstance = size(trainingSet,3); %ѵ��������  
    numTestInstance = size(testingSet,3);   %����������  
    height = size(trainingSet,1);          %ͼ��߶�  
    width = size(trainingSet,2);         %ͼ����  
    perClassTrainLen = numTrainInstance/numClass;%ÿ������ѵ��������  
    perClassTestLen = numTestInstance/numClass;%ÿ�����Ĳ���������  
     
    numShape = 3;%ѵ����ʱ��ͶӰ������ά�����ϣ�numShape<height  
    projection = zeros(width,numShape);  
    allprojectionFace = zeros(height,numShape,numTrainInstance);  
      
      
      
    %����Ϊѵ��  
    meanFace = mean(trainingSet,3);  
    %��Э�������GT  
    CovG = zeros(width,width);  
    for n=1:numTrainInstance  
        disToMean = trainingSet(:,:,n)-meanFace;  
        CovG = CovG+disToMean'*disToMean;  
    end  
    CovG = CovG/numTrainInstance;  
  
    %������ֵ����������  
    [eigenFace,eigenValue] = svd(CovG);  
      
    if(nargin==4)  
        if(thresthold<1)%���������ֵΪ<1��Ϊ��Ҫ��̬���d��ȡֵ  
            %disp('u set thresthold');  
            sumEigenValue = 0;  
            tmp = 0;  
            for xi=1:size(eigenValue,2)  
                sumEigenValue = sumEigenValue+eigenValue(xi,xi);  
            end  
            for xi=1:size(eigenValue,2)  
                tmp = tmp+eigenValue(xi,xi);  
                if(tmp/sumEigenValue>thresthold)  
                    break;  
                end  
            end;  
            numShape = xi;    
        else %������Ϊ���õ���ֵΪd��ֵ����ͶӰ������ά��  
            %disp('u set the d value...');  
            numShape = thresthold;  
        end  
              
        projection = zeros(width,numShape);  
        allprojectionFace = zeros(height,numShape,numTrainInstance);  
      
    end  
  
    %��ͶӰ����projection,��ͼ����������������ͼ��  
    for k=1:numShape  
        projection(:,k) = eigenFace(:,k);  
    end  
  
    %����ÿ��ѵ������ͶӰ�������  
    for inum = 1:numTrainInstance  
        allprojectionFace(:,:,inum) = trainingSet(:,:,inum)*projection;  
    end;  
      
    %����Ϊ����  
    right = 0;  
     
    for x=1:numTestInstance  
        afterProjection = testingSet(:,:,x)*projection;     
        error = zeros(numTrainInstance,1);  
        for i=1:numTrainInstance  
            %����ͶӰ���ͼ����󵽸������ͼ������ľ���  
            miss = afterProjection -allprojectionFace(:,:,i);  
            for j=1:size(miss,2)  
                error(i) =error(i)+ norm(miss(:,j));  
            end  
        end;  
         
        [errorS,errorIndex] = sort(error);  %�Ծ����������  
        class = floor((errorIndex(1)-1)/perClassTrainLen)+1;%��ͼ��ֵ�������С�������ȥ,Ԥ������  
          
        oriclass =  floor((x-1)/perClassTestLen)+1 ; %ʵ�ʵ����  
        if(class == oriclass)  
            right = right+1;  
        end  
    end  
      
    accuracy = right/numTestInstance;  
      
end