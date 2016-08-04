%--------------------------------------------------------------------------  
% ����˵����2DPCA�㷨  
% ���ղ���  
%   trainingSet:ѵ�����ݼ���height*width*num�ľ���  
%   testingSet:�������ݼ���3ά����  
%   numClass:ѵ�������Ͳ���������ӵ�е������  
%   no_dims:��ֵ��������������������ѡ�񣬼Ⱦ���dȡֵ��  
%              �����ݵ�no_dimsΪС������0.95���Զ�������ֵ���㣬d,q��ȡֵ��  
%              �����ݵ�no_dimsΪ>=1����������30,�����d=30,numShape=d  
%     
% ��������  
%     accuracy:����׼ȷ��  
%     right:��ȷ�����������Ŀ  
%   
    load('cell_orldata_label.mat')  
    cell_dataSet = cell_orldata;
    no_dims = 2;
%function [project_cell_dataSet, projection]=TDPCA(cell_dataSet,no_dims)  
     
    num_dataSet = size(cell_dataSet,2); %ѵ��������  
    height = size(cell_dataSet{1},1);          %ͼ��߶�  
    width = size(cell_dataSet{1},2);         %ͼ����
    
    % ��cell���� ת���� 3ά����
    mat_dataSet_3d = zeros(height,width,num_dataSet);
    for i = 1:num_dataSet
        mat_dataSet_3d(:,:,i) = cell_dataSet{i};
    end      
      
    meanFace = mean(mat_dataSet_3d,3);  
    %��Э�������GT  
    CovG = zeros(width,width);  
    for n=1:num_dataSet  
        disToMean = mat_dataSet_3d(:,:,n)-meanFace;  
        CovG = CovG+disToMean'*disToMean;  
    end  
    CovG = CovG/num_dataSet;  
  
    %������ֵ����������
    [eigenFace,eigenValue] = svd(CovG);
    
    projection = zeros(width,no_dims);      % ͶӰ����
      
    %��ͶӰ����projection,��ͼ����������������ͼ��  
    for k=1:no_dims  
        projection(:,k) = eigenFace(:,k);  
    end
    
    project_dataSet_3d = zeros(height,no_dims,num_dataSet);        % ͶӰ�����ά����
    %����ÿ��ѵ������ͶӰ�������
    for inum = 1:num_dataSet  
        project_dataSet_3d(:,:,inum) = mat_dataSet_3d(:,:,inum)*projection;  
    end
    
    % ��ά����ת����cell
    for i = 1:num_dataSet
        project_cell_dataSet{i} = project_dataSet_3d(:,:,i);
    end