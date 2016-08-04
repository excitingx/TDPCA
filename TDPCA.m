%--------------------------------------------------------------------------  
% 函数说明：2DPCA算法  
% 接收参数  
%   trainingSet:训练数据集，height*width*num的矩阵  
%   testingSet:测试数据集，3维矩阵  
%   numClass:训练样本和测试样本所拥有的类别数  
%   no_dims:阈值，决定特征向量个数的选择，既决定d取值，  
%              若传递的no_dims为小数，如0.95，自动根据阈值计算，d,q的取值；  
%              若传递的no_dims为>=1的整数，如30,则会设d=30,numShape=d  
%     
% 函数返回  
%     accuracy:分类准确率  
%     right:正确分类的样本数目  
%   
    load('cell_orldata_label.mat')  
    cell_dataSet = cell_orldata;
    no_dims = 2;
%function [project_cell_dataSet, projection]=TDPCA(cell_dataSet,no_dims)  
     
    num_dataSet = size(cell_dataSet,2); %训练样本数  
    height = size(cell_dataSet{1},1);          %图像高度  
    width = size(cell_dataSet{1},2);         %图像宽度
    
    % 将cell类型 转换成 3维矩阵
    mat_dataSet_3d = zeros(height,width,num_dataSet);
    for i = 1:num_dataSet
        mat_dataSet_3d(:,:,i) = cell_dataSet{i};
    end      
      
    meanFace = mean(mat_dataSet_3d,3);  
    %求协方差矩阵GT  
    CovG = zeros(width,width);  
    for n=1:num_dataSet  
        disToMean = mat_dataSet_3d(:,:,n)-meanFace;  
        CovG = CovG+disToMean'*disToMean;  
    end  
    CovG = CovG/num_dataSet;  
  
    %求特征值和特征向量
    [eigenFace,eigenValue] = svd(CovG);
    
    projection = zeros(width,no_dims);      % 投影矩阵
      
    %求投影向量projection,即图像的特征矩阵或特征图像  
    for k=1:no_dims  
        projection(:,k) = eigenFace(:,k);  
    end
    
    project_dataSet_3d = zeros(height,no_dims,num_dataSet);        % 投影后的三维矩阵
    %保存每个训练样本投影后的特征
    for inum = 1:num_dataSet  
        project_dataSet_3d(:,:,inum) = mat_dataSet_3d(:,:,inum)*projection;  
    end
    
    % 三维矩阵转换成cell
    for i = 1:num_dataSet
        project_cell_dataSet{i} = project_dataSet_3d(:,:,i);
    end